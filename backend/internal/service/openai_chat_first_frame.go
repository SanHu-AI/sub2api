package service

import (
	"crypto/rand"
	"encoding/hex"
	"net/http"
	"strings"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/apicompat"
	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// 首帧快返（first-frame fastpath）使用的 gin context 键。
//
// 首帧快返在转发上游之前就向客户端下发一个内容为空的 assistant 开场帧，目的
// 是让下游立刻拿到"首字"（提交 200 + SSE 响应头 + 一个 delta 为空的
// chat.completion.chunk），把上游排队、TLS 往返和模型思考时间从客户端感知的
// TTFT 里挪走。
//
// 它必须同时满足三条约束：
//  1. 不改变模型输出内容——开场帧 delta 只有 role=assistant 和空 content；
//  2. 不污染 failover——首帧字节记入 openAIChatFirstFrameBytesKey，并由
//     OpenAICompactKeepaliveAdjustedWrittenSize 扣除，所以"是否已向客户端写出
//     语义响应"的判定仍然为否，上游 429/5xx 依旧可以换号重试；
//  3. 不重复下发 role——上游真实开场帧由 SentRole（Responses 桥接路径）或
//     stripInjectedChatRoleFromSSELine（原生 CC 直转路径）去重。
const (
	openAIChatFirstFrameKey      = "openai_chat_first_frame"
	openAIChatFirstFrameIDKey    = "openai_chat_first_frame_id"
	openAIChatFirstFrameBytesKey = "openai_chat_first_frame_bytes"
)

// OpenAIChatFirstFrameInjected 报告本次请求是否已经下发过快返首帧。
func OpenAIChatFirstFrameInjected(c *gin.Context) bool {
	if c == nil {
		return false
	}
	value, ok := c.Get(openAIChatFirstFrameKey)
	if !ok {
		return false
	}
	injected, _ := value.(bool)
	return injected
}

// OpenAIChatFirstFrameID 返回快返首帧使用的 chatcmpl ID。后续转换路径复用同一
// 个 ID，避免同一条流里出现两个不同的 completion ID。
func OpenAIChatFirstFrameID(c *gin.Context) string {
	if c == nil {
		return ""
	}
	value, ok := c.Get(openAIChatFirstFrameIDKey)
	if !ok {
		return ""
	}
	id, _ := value.(string)
	return id
}

// WriteOpenAIChatFirstFrame 立即向客户端下发开场帧并提交 SSE 响应头。
//
// 返回 false 表示本次没有下发（已写过响应 / 已下发过 / 写出失败），调用方应按
// 原路径继续处理。函数幂等：同一请求重复调用只会下发一次。
//
// 只应在确认请求为流式且响应头尚未提交时调用；非流式请求没有"提前返回部分
// 响应"的合法载体，不应使用。
func WriteOpenAIChatFirstFrame(c *gin.Context, model string) bool {
	if c == nil || c.Writer == nil {
		return false
	}
	if c.Writer.Written() || OpenAIChatFirstFrameInjected(c) {
		return false
	}

	empty := ""
	id := generateOpenAIChatFirstFrameID()
	chunk := apicompat.ChatCompletionsChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Model:   strings.TrimSpace(model),
		Choices: []apicompat.ChatChunkChoice{{
			Index:        0,
			Delta:        apicompat.ChatDelta{Role: "assistant", Content: &empty},
			FinishReason: nil,
		}},
	}
	sse, err := apicompat.ChatChunkToSSE(chunk)
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
	c.Set(openAIChatFirstFrameBytesKey, n)
	c.Set(openAIChatFirstFrameIDKey, id)
	c.Set(openAIChatFirstFrameKey, true)
	if flusher, ok := c.Writer.(http.Flusher); ok {
		flusher.Flush()
	}
	return true
}

// openAIChatFirstFrameBytes 返回已下发首帧占用的字节数。
func openAIChatFirstFrameBytes(c *gin.Context) int {
	if c == nil {
		return 0
	}
	value, ok := c.Get(openAIChatFirstFrameBytesKey)
	if !ok {
		return 0
	}
	bytes, _ := value.(int)
	if bytes < 0 {
		return 0
	}
	return bytes
}

// stripInjectedChatRoleFromSSELine 在已下发首帧的前提下，剔除上游开场 chunk 里
// 的 delta.role，避免客户端收到两次 assistant 角色帧。
//
// roleStripped 由调用方持有并在整条流上复用：只处理第一个带 role 的 chunk，
// 之后的 chunk 原样透传。解析失败或不是 JSON 数据行时返回原始行。
func stripInjectedChatRoleFromSSELine(line string, roleStripped *bool) string {
	if roleStripped == nil || *roleStripped {
		return line
	}
	payload, ok := extractOpenAISSEDataLine(line)
	if !ok {
		return line
	}
	trimmed := strings.TrimSpace(payload)
	if trimmed == "" || trimmed == "[DONE]" {
		return line
	}
	if !gjson.Get(payload, "choices.0.delta.role").Exists() {
		return line
	}
	updated, err := sjson.DeleteBytes([]byte(payload), "choices.0.delta.role")
	if err != nil {
		return line
	}
	*roleStripped = true
	return "data: " + string(updated) + "\n\n"
}

// generateOpenAIChatFirstFrameID 生成与上游同形的 chatcmpl ID。
func generateOpenAIChatFirstFrameID() string {
	b := make([]byte, 12)
	_, _ = rand.Read(b)
	return "chatcmpl-" + hex.EncodeToString(b)
}
