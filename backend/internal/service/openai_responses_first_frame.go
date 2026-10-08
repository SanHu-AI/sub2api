package service

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"net/http"
	"strings"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/apicompat"
	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
	"go.uber.org/zap"
)

// /v1/responses 首帧快返（first-frame fastpath）使用的 gin context 键。
//
// 与 Chat Completions 首帧快返同源：在转发上游之前就向客户端下发
// event: response.created 并提交 200 + SSE 响应头，把上游排队与模型思考时间从
// 客户端感知的 TTFT 里挪走。
//
// 相比 chat 路径，responses 多一条硬约束：**response.id 必须可用**。
// 客户端（Codex 等）会把 response.created 里的 id 当作下一轮的
// previous_response_id；而首帧下发时上游真实 id 还不存在，只能先编一个。因此：
//  1. 编造 id 带固定前缀 openAIResponsesFirstFrameAliasPrefix，形态仍满足上游
//     id 正则（^resp_[A-Za-z0-9_-]{1,256}$），不会被入口判为非法；
//  2. 流内拿到上游真实 id 后，由 openAIResponsesFirstFrameBindRealID 写入
//     Redis 别名映射（编造 id → 真实 id）；
//  3. 后续请求带 previous_response_id=编造 id 时，handler 在归属校验之前用
//     ResolveOpenAIResponsesFirstFrameAlias 翻译回真实 id，续链才不会断。
//
// 另外两条与 chat 路径一致的约束：
//  4. 首帧字节记入 openAIResponsesFirstFrameBytesKey，并由
//     OpenAICompactKeepaliveAdjustedWrittenSize 扣除，保证"是否已写出语义响应"
//     的判定仍为否，上游 429/5xx 依旧可以换号重试；
//  5. 流内所有带 response id 的事件统一改写为编造 id，避免同一条流里出现两个
//     不同的 response id。
const (
	openAIResponsesFirstFrameKey      = "openai_responses_first_frame"
	openAIResponsesFirstFrameAliasKey = "openai_responses_first_frame_alias"
	openAIResponsesFirstFrameBytesKey = "openai_responses_first_frame_bytes"
	openAIResponsesFirstFrameRealIDKey = "openai_responses_first_frame_real_id"
)

// openAIResponsesFirstFrameAliasPrefix 是首帧编造 response id 的前缀。
const openAIResponsesFirstFrameAliasPrefix = "resp_s2ff_"

// openAIResponsesFirstFrameAliasTTL 是别名映射的有效期。与 reasoning 缓存一致
// 取 7 天，覆盖跨天恢复的 Codex 会话。
const openAIResponsesFirstFrameAliasTTL = 7 * 24 * time.Hour

// openAIResponsesAliasRedisTimeout 是别名读写的有界预算：首帧路径不能因为
// Redis 抖动而拖慢流式首字。
const openAIResponsesAliasRedisTimeout = 3 * time.Second

// ResponsesFirstFrameEnabled 报告是否开启流式 /v1/responses 首帧快返。
//
// 运行时设置（setting key openai_responses_first_frame_enabled，管理员在设置页
// 控制），读取走 SettingService 的进程内缓存（60s TTL），热路径无 DB 查询。
// settingService 未注入时一律返回 false，即保持关闭。
func (s *OpenAIGatewayService) ResponsesFirstFrameEnabled(ctx context.Context) bool {
	if s == nil || s.settingService == nil {
		return false
	}
	return s.settingService.IsOpenAIResponsesFirstFrameEnabled(ctx)
}

// OpenAIResponsesFirstFrameInjected 报告本次请求是否已经下发过快返首帧。
func OpenAIResponsesFirstFrameInjected(c *gin.Context) bool {
	if c == nil {
		return false
	}
	value, ok := c.Get(openAIResponsesFirstFrameKey)
	if !ok {
		return false
	}
	injected, _ := value.(bool)
	return injected
}

// OpenAIResponsesFirstFrameAliasID 返回首帧使用的编造 response id。流内所有后
// 续事件的 response id 都必须改写为它，保证客户端只看到一个 id。
func OpenAIResponsesFirstFrameAliasID(c *gin.Context) string {
	if c == nil {
		return ""
	}
	value, ok := c.Get(openAIResponsesFirstFrameAliasKey)
	if !ok {
		return ""
	}
	id, _ := value.(string)
	return id
}

// openAIResponsesFirstFrameBytes 返回已下发首帧占用的字节数。
func openAIResponsesFirstFrameBytes(c *gin.Context) int {
	if c == nil {
		return 0
	}
	value, ok := c.Get(openAIResponsesFirstFrameBytesKey)
	if !ok {
		return 0
	}
	written, _ := value.(int)
	if written < 0 {
		return 0
	}
	return written
}

// WriteOpenAIResponsesFirstFrame 立即向客户端下发 response.created 并提交 SSE
// 响应头。
//
// 返回 false 表示本次没有下发（已写过响应 / 已下发过 / 序列化或写出失败），调
// 用方应按原路径继续处理。函数幂等：同一请求重复调用只会下发一次。
//
// 只应在确认请求为流式且响应头尚未提交时调用。
func WriteOpenAIResponsesFirstFrame(c *gin.Context, model string) bool {
	if c == nil || c.Writer == nil {
		return false
	}
	if c.Writer.Written() || OpenAIResponsesFirstFrameInjected(c) {
		return false
	}

	aliasID := generateOpenAIResponsesFirstFrameAlias()
	event := apicompat.ResponsesStreamEvent{
		Type:           "response.created",
		SequenceNumber: 0,
		Response: &apicompat.ResponsesResponse{
			ID:        aliasID,
			Object:    "response",
			CreatedAt: time.Now().Unix(),
			Model:     strings.TrimSpace(model),
			Status:    "in_progress",
			Output:    []apicompat.ResponsesOutput{},
		},
	}
	sse, err := apicompat.ResponsesEventToSSE(event)
	if err != nil {
		return false
	}

	c.Writer.Header().Set("Content-Type", "text/event-stream")
	c.Writer.Header().Set("Cache-Control", "no-cache")
	c.Writer.Header().Set("Connection", "keep-alive")
	c.Writer.Header().Set("X-Accel-Buffering", "no")
	c.Writer.WriteHeader(http.StatusOK)

	n, err := c.Writer.Write([]byte(sse))
	if err != nil {
		return false
	}
	c.Set(openAIResponsesFirstFrameBytesKey, n)
	c.Set(openAIResponsesFirstFrameAliasKey, aliasID)
	c.Set(openAIResponsesFirstFrameKey, true)
	if flusher, ok := c.Writer.(http.Flusher); ok {
		flusher.Flush()
	}
	return true
}

// generateOpenAIResponsesFirstFrameAlias 生成带固定前缀的编造 response id。
// 16 字节随机量 + 固定前缀，既满足上游 id 形态，也足够避免碰撞。
func generateOpenAIResponsesFirstFrameAlias() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	return openAIResponsesFirstFrameAliasPrefix + hex.EncodeToString(b)
}

// IsOpenAIResponsesFirstFrameAlias 判断 id 是否为首帧快返编造的 response id。
func IsOpenAIResponsesFirstFrameAlias(id string) bool {
	id = strings.TrimSpace(id)
	return strings.HasPrefix(id, openAIResponsesFirstFrameAliasPrefix)
}

// openAIResponsesFirstFrameBindRealID 记录上游真实 response id 并落 Redis 别名
// 映射。
//
// 只在首帧已下发时生效；同一个真实 id 只写一次（流内每个事件都带 response.id，
// 不去重会把 Redis 写爆）。真实 id 发生变化时覆盖：failover 场景下首个账号可能
// 已经吐出 response.id 后才失败，换号成功的那个 id 才是客户端真正续接需要的。
//
// Redis 不可用时仅记录到 context（本次流的 id 一致性仍正确），续链翻译会失效但
// 不影响本次响应。
func (s *OpenAIGatewayService) openAIResponsesFirstFrameBindRealID(c *gin.Context, realID string) {
	if s == nil || c == nil {
		return
	}
	realID = strings.TrimSpace(realID)
	aliasID := OpenAIResponsesFirstFrameAliasID(c)
	if aliasID == "" || realID == "" || realID == aliasID {
		return
	}
	if bound, ok := c.Get(openAIResponsesFirstFrameRealIDKey); ok {
		if boundID, _ := bound.(string); boundID == realID {
			return
		}
	}
	c.Set(openAIResponsesFirstFrameRealIDKey, realID)
	if s.cache == nil {
		return
	}
	requestCtx := c.Request.Context()
	if requestCtx == nil {
		requestCtx = context.Background()
	}
	// 客户端可能在收到终止事件后立刻断开并取消 ctx，这里用 WithoutCancel +
	// 有界预算，保证映射一定有机会落盘。
	bindCtx, cancel := context.WithTimeout(context.WithoutCancel(requestCtx), openAIResponsesAliasRedisTimeout)
	defer cancel()
	if err := s.cache.SetResponseIDAlias(bindCtx, aliasID, realID, openAIResponsesFirstFrameAliasTTL); err != nil {
		logOpenAIResponsesFirstFrameAliasWarn(aliasID, err)
	}
}

// ResolveOpenAIResponsesFirstFrameAlias 把客户端带来的编造 id 翻译回上游真实
// response id。
//
// 只在 id 确为首帧编造、且命中 Redis 映射时返回真实 id；其余情况返回 false，
// 调用方应保留原 id（交由上游判定或走既有的 previous_response_not_found 恢复）。
func (s *OpenAIGatewayService) ResolveOpenAIResponsesFirstFrameAlias(ctx context.Context, id string) (string, bool) {
	id = strings.TrimSpace(id)
	if !IsOpenAIResponsesFirstFrameAlias(id) || s == nil || s.cache == nil {
		return "", false
	}
	if ctx == nil {
		ctx = context.Background()
	}
	resolveCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), openAIResponsesAliasRedisTimeout)
	defer cancel()
	realID, err := s.cache.GetResponseIDAlias(resolveCtx, id)
	if err != nil {
		if err != ErrResponseIDAliasNotFound {
			logOpenAIResponsesFirstFrameAliasWarn(id, err)
		}
		return "", false
	}
	realID = strings.TrimSpace(realID)
	if realID == "" {
		return "", false
	}
	// 翻译结果必须是合法 response id：Anthropic native 等路径的"上游真实 id"可能是
	// msg_ 形态，透传给上游会被判成 message_id 直接 400。这种情形保留编造 id 更
	// 安全（形态合法，最多让上游报 previous_response_not_found 并走既有恢复）。
	if ClassifyOpenAIPreviousResponseIDKind(realID) != OpenAIPreviousResponseIDKindResponseID {
		return "", false
	}
	return realID, true
}

// rewriteResponsesFirstFrameIDBytes 把单个 Responses 事件 JSON 里的 response id
// 改写为首帧编造 id，并在首次拿到上游真实 id 时落别名映射。
//
// 用于直接操作事件 JSON（而非 SSE 文本行）的转发路径，例如 Anthropic native。
func (s *OpenAIGatewayService) rewriteResponsesFirstFrameIDBytes(c *gin.Context, payload []byte) []byte {
	if s == nil || c == nil || len(payload) == 0 {
		return payload
	}
	aliasID := OpenAIResponsesFirstFrameAliasID(c)
	if aliasID == "" {
		return payload
	}
	realID := strings.TrimSpace(gjson.GetBytes(payload, "response.id").String())
	if realID == "" || realID == aliasID {
		return payload
	}
	s.openAIResponsesFirstFrameBindRealID(c, realID)
	updated, err := sjson.SetBytes(payload, "response.id", aliasID)
	if err != nil {
		return payload
	}
	return updated
}

// rewriteOpenAIResponsesFirstFrameIDLine 把 Responses SSE 行里的 response id 统
// 一改写为首帧编造 id，并返回行中原本携带的上游真实 id（若无则为空串）。
//
// 调用方应把返回的真实 id 交给 openAIResponsesFirstFrameBindRealID 落映射。解
// 析失败、非 JSON 数据行、或 id 已是编造 id 时原样返回。
func rewriteOpenAIResponsesFirstFrameIDLine(line string, aliasID string) (string, string) {
	if aliasID == "" {
		return line, ""
	}
	payload, ok := extractOpenAISSEDataLine(line)
	if !ok {
		return line, ""
	}
	trimmed := strings.TrimSpace(payload)
	if trimmed == "" || trimmed == "[DONE]" {
		return line, ""
	}
	// response 类事件把 id 放在 response.id；少数上游（或终止事件）放在顶层 id。
	idPath := "response.id"
	if !gjson.Get(payload, "response.id").Exists() {
		if !gjson.Get(payload, "id").Exists() {
			return line, ""
		}
		idPath = "id"
	}
	realID := strings.TrimSpace(gjson.Get(payload, idPath).String())
	if realID == "" || realID == aliasID {
		return line, ""
	}
	updated, err := sjson.SetBytes([]byte(payload), idPath, aliasID)
	if err != nil {
		return line, realID
	}
	// 与 replaceModelInSSELine 保持同一约定：只返回 "data: " 前缀的单行，行尾换行
	// 由写出侧统一补，避免多出一个空行触发额外的 SSE 事件分派。
	return "data: " + string(updated), realID
}

// logOpenAIResponsesFirstFrameAliasWarn 记录别名映射读写失败。首帧路径不能因为
// Redis 抖动影响流式响应，所以只告警不中断。
func logOpenAIResponsesFirstFrameAliasWarn(responseID string, err error) {
	if err == nil {
		return
	}
	logger.L().Warn(
		"openai.responses_first_frame_alias_failed",
		zap.String("response_id", responseID),
		zap.Error(err),
	)
}
