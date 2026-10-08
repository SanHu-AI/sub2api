package service

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

// stubFirstFrameAliasCache 只实现别名映射两个方法，其余方法走内嵌接口（未使用）。
type stubFirstFrameAliasCache struct {
	GatewayCache
	aliases  map[string]string
	setCalls []string
}

func newStubFirstFrameAliasCache() *stubFirstFrameAliasCache {
	return &stubFirstFrameAliasCache{aliases: map[string]string{}}
}

func (c *stubFirstFrameAliasCache) SetResponseIDAlias(_ context.Context, aliasID string, realID string, _ time.Duration) error {
	c.setCalls = append(c.setCalls, aliasID)
	c.aliases[aliasID] = realID
	return nil
}

func (c *stubFirstFrameAliasCache) GetResponseIDAlias(_ context.Context, aliasID string) (string, error) {
	if realID, ok := c.aliases[aliasID]; ok {
		return realID, nil
	}
	return "", ErrResponseIDAliasNotFound
}

func newResponsesFirstFrameTestContext(t *testing.T) (*gin.Context, *httptest.ResponseRecorder) {
	t.Helper()
	gin.SetMode(gin.TestMode)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
	return c, rec
}

func responsesFirstFrameDataPayload(t *testing.T, body string) string {
	t.Helper()
	idx := strings.Index(body, "data: ")
	require.GreaterOrEqual(t, idx, 0)
	return body[idx+len("data: "):]
}

// 首帧必须是 response.created，id 为网关编造的 resp_s2ff_ 前缀，且提交标准 SSE 头。
func TestWriteOpenAIResponsesFirstFrame_WritesResponseCreated(t *testing.T) {
	c, rec := newResponsesFirstFrameTestContext(t)

	require.True(t, WriteOpenAIResponsesFirstFrame(c, "gpt-5"))
	require.Equal(t, http.StatusOK, rec.Code)
	require.Equal(t, "text/event-stream", rec.Header().Get("Content-Type"))
	require.Equal(t, "no-cache", rec.Header().Get("Cache-Control"))
	require.Equal(t, "no", rec.Header().Get("X-Accel-Buffering"))

	body := rec.Body.String()
	require.Contains(t, body, "event: response.created\n")
	require.Contains(t, body, "\n\n")

	aliasID := OpenAIResponsesFirstFrameAliasID(c)
	require.True(t, IsOpenAIResponsesFirstFrameAlias(aliasID))
	require.True(t, OpenAIResponsesFirstFrameInjected(c))

	payload := responsesFirstFrameDataPayload(t, body)
	require.Equal(t, "response.created", gjson.Get(payload, "type").String())
	require.Equal(t, aliasID, gjson.Get(payload, "response.id").String())
	require.Equal(t, "response", gjson.Get(payload, "response.object").String())
	require.Equal(t, "gpt-5", gjson.Get(payload, "response.model").String())
	require.Equal(t, "in_progress", gjson.Get(payload, "response.status").String())
	// 空 output：首帧不产生任何内容 token，不影响上下文。
	require.Empty(t, gjson.Get(payload, "response.output").Array(), "首帧不得携带任何 output")
	// 编造 id 必须仍被上游 id 形态接受，否则入口会直接判为非法。
	require.Equal(t, OpenAIPreviousResponseIDKindResponseID, ClassifyOpenAIPreviousResponseIDKind(aliasID))

	// 幂等：同一请求不会重复下发。
	written := rec.Body.Len()
	require.False(t, WriteOpenAIResponsesFirstFrame(c, "gpt-5"))
	require.Equal(t, written, rec.Body.Len())
}

// 首帧字节必须从"是否已写出语义响应"的口径里扣除，否则上游 429/5xx 会因为首帧
// 已经发出而不再换号重试。
func TestOpenAIAdjustedWrittenSizeExcludesResponsesFirstFrame(t *testing.T) {
	c, _ := newResponsesFirstFrameTestContext(t)

	before := OpenAICompactKeepaliveAdjustedWrittenSize(c)
	require.True(t, WriteOpenAIResponsesFirstFrame(c, "gpt-5"))
	require.Equal(t, before, OpenAICompactKeepaliveAdjustedWrittenSize(c),
		"仅有首帧字节时口径必须不变，否则 failover 换号会被误判放弃")

	_, err := c.Writer.Write([]byte("data: real\n\n"))
	require.NoError(t, err)
	require.Equal(t, len("data: real\n\n"), OpenAICompactKeepaliveAdjustedWrittenSize(c))
}

// 流内事件的 response.id 必须统一改写为编造 id，并返回上游真实 id 供落映射。
func TestRewriteOpenAIResponsesFirstFrameIDLine(t *testing.T) {
	const alias = "resp_s2ff_abc"

	// response.id 形态：改写并回传真实 id。
	line, realID := rewriteOpenAIResponsesFirstFrameIDLine(
		`data: {"type":"response.created","response":{"id":"resp_upstream_1","status":"in_progress"}}`, alias)
	require.Equal(t, "resp_upstream_1", realID)
	require.NotContains(t, line, "\n", "改写结果必须是单行，行尾换行由写出侧统一补")
	require.Equal(t, alias, gjson.Get(line, "response.id").String())
	require.Equal(t, "response.created", gjson.Get(line, "type").String())

	// 顶层 id 形态（终止事件等）：同样改写。
	line, realID = rewriteOpenAIResponsesFirstFrameIDLine(
		`data: {"type":"response.completed","id":"resp_upstream_2"}`, alias)
	require.Equal(t, "resp_upstream_2", realID)
	require.Equal(t, alias, gjson.Get(line, "id").String())

	// 已经是编造 id：原样返回，不回传真实 id。
	line, realID = rewriteOpenAIResponsesFirstFrameIDLine(`data: {"response":{"id":"`+alias+`"}}`, alias)
	require.Equal(t, "", realID)
	require.Contains(t, line, alias)

	// 无 id 的普通输出事件、终止行、非数据行：原样透传。
	for _, raw := range []string{
		`data: {"type":"response.output_text.delta","delta":"hi"}`,
		`data: [DONE]`,
		`event: response.created`,
		``,
	} {
		out, id := rewriteOpenAIResponsesFirstFrameIDLine(raw, alias)
		require.Equal(t, raw, out)
		require.Equal(t, "", id)
	}

	// 未下发首帧（alias 为空）时不得改动任何行。
	raw := `data: {"response":{"id":"resp_upstream_1"}}`
	out, id := rewriteOpenAIResponsesFirstFrameIDLine(raw, "")
	require.Equal(t, raw, out)
	require.Equal(t, "", id)
}

// 拿到上游真实 id 后必须落一次 Redis 别名映射，重复调用不重复写。
func TestOpenAIResponsesFirstFrameBindRealID(t *testing.T) {
	c, _ := newResponsesFirstFrameTestContext(t)
	cache := newStubFirstFrameAliasCache()
	svc := &OpenAIGatewayService{cache: cache}

	require.True(t, WriteOpenAIResponsesFirstFrame(c, "gpt-5"))
	aliasID := OpenAIResponsesFirstFrameAliasID(c)

	svc.openAIResponsesFirstFrameBindRealID(c, "resp_upstream_1")
	svc.openAIResponsesFirstFrameBindRealID(c, "resp_upstream_1")
	require.Equal(t, []string{aliasID}, cache.setCalls, "同一条流只应写一次映射")
	require.Equal(t, "resp_upstream_1", cache.aliases[aliasID])

	// failover：首个账号吐出 id 后失败，换号成功的 id 必须覆盖旧映射，
	// 否则客户端续接会拿到一个上游并不认识的 id。
	svc.openAIResponsesFirstFrameBindRealID(c, "resp_upstream_2")
	require.Equal(t, []string{aliasID, aliasID}, cache.setCalls)
	require.Equal(t, "resp_upstream_2", cache.aliases[aliasID])

	// 未下发首帧 / 空 id / 真实 id 等于编造 id：都不写。
	plain, _ := newResponsesFirstFrameTestContext(t)
	svc.openAIResponsesFirstFrameBindRealID(plain, "resp_upstream_1")
	svc.openAIResponsesFirstFrameBindRealID(c, "")
	svc.openAIResponsesFirstFrameBindRealID(c, aliasID)
	require.Len(t, cache.setCalls, 2)
}

// 入口翻译：只接受编造 id 且命中映射；非法形态的真实 id 不能透传给上游。
func TestResolveOpenAIResponsesFirstFrameAlias(t *testing.T) {
	ctx := context.Background()
	cache := newStubFirstFrameAliasCache()
	svc := &OpenAIGatewayService{cache: cache}

	aliasID := generateOpenAIResponsesFirstFrameAlias()
	cache.aliases[aliasID] = "resp_upstream_1"

	realID, ok := svc.ResolveOpenAIResponsesFirstFrameAlias(ctx, aliasID)
	require.True(t, ok)
	require.Equal(t, "resp_upstream_1", realID)

	// Anthropic native 等路径的"真实 id"可能是 msg_ 形态，透传会让上游 400，
	// 必须拒绝翻译（保留编造 id 走既有 previous_response_not_found 恢复）。
	msgAlias := generateOpenAIResponsesFirstFrameAlias()
	cache.aliases[msgAlias] = "msg_0123456789"
	_, ok = svc.ResolveOpenAIResponsesFirstFrameAlias(ctx, msgAlias)
	require.False(t, ok)

	// 非编造 id、未命中映射、无 cache 注入：一律不翻译。
	_, ok = svc.ResolveOpenAIResponsesFirstFrameAlias(ctx, "resp_upstream_1")
	require.False(t, ok)
	_, ok = svc.ResolveOpenAIResponsesFirstFrameAlias(ctx, generateOpenAIResponsesFirstFrameAlias())
	require.False(t, ok)
	_, ok = (&OpenAIGatewayService{}).ResolveOpenAIResponsesFirstFrameAlias(ctx, aliasID)
	require.False(t, ok)
}

// 事件 JSON 形态的改写（Anthropic native 路径）必须同步落映射并换 id。
func TestRewriteResponsesFirstFrameIDBytes(t *testing.T) {
	c, _ := newResponsesFirstFrameTestContext(t)
	cache := newStubFirstFrameAliasCache()
	svc := &OpenAIGatewayService{cache: cache}

	require.True(t, WriteOpenAIResponsesFirstFrame(c, "gpt-5"))
	aliasID := OpenAIResponsesFirstFrameAliasID(c)

	out := svc.rewriteResponsesFirstFrameIDBytes(c,
		[]byte(`{"type":"response.created","response":{"id":"resp_upstream_1"}}`))
	require.Equal(t, aliasID, gjson.GetBytes(out, "response.id").String())
	require.Equal(t, "resp_upstream_1", cache.aliases[aliasID])

	// 无 response.id 的事件原样返回。
	raw := []byte(`{"type":"response.output_text.delta","delta":"hi"}`)
	require.Equal(t, raw, svc.rewriteResponsesFirstFrameIDBytes(c, raw))
}

// 开关由运行时设置控制：key 缺失或显式关闭时必须保持关闭；settingService 未注入
// 也必须保持关闭。
func TestResponsesFirstFrameEnabled_RuntimeSetting(t *testing.T) {
	ctx := context.Background()
	repo := &gatewayTTLSettingRepo{data: map[string]string{}}
	gatewayForwardingCache.Store(&cachedGatewayForwardingSettings{})
	svc := &OpenAIGatewayService{settingService: NewSettingService(repo, &config.Config{})}

	require.False(t, svc.ResponsesFirstFrameEnabled(ctx), "key 缺失时必须默认关闭")

	repo.data[SettingKeyOpenAIResponsesFirstFrameEnabled] = "false"
	gatewayForwardingCache.Store(&cachedGatewayForwardingSettings{})
	require.False(t, svc.ResponsesFirstFrameEnabled(ctx))

	repo.data[SettingKeyOpenAIResponsesFirstFrameEnabled] = "true"
	gatewayForwardingCache.Store(&cachedGatewayForwardingSettings{})
	require.True(t, svc.ResponsesFirstFrameEnabled(ctx))

	require.False(t, (&OpenAIGatewayService{}).ResponsesFirstFrameEnabled(ctx))
}
