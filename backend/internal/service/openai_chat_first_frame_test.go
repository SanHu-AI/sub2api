package service

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func newFirstFrameTestContext(t *testing.T) (*gin.Context, *httptest.ResponseRecorder) {
	t.Helper()
	gin.SetMode(gin.TestMode)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	return c, rec
}

// 首帧快返必须只下发一次内容为空的 assistant 开场帧，并提交标准 SSE 响应头。
func TestWriteOpenAIChatFirstFrame_WritesEmptyAssistantChunk(t *testing.T) {
	c, rec := newFirstFrameTestContext(t)

	require.True(t, WriteOpenAIChatFirstFrame(c, "gpt-5"))
	require.Equal(t, http.StatusOK, rec.Code)
	require.Equal(t, "text/event-stream", rec.Header().Get("Content-Type"))
	require.Equal(t, "no-cache", rec.Header().Get("Cache-Control"))
	require.Equal(t, "no", rec.Header().Get("X-Accel-Buffering"))

	body := rec.Body.String()
	require.Contains(t, body, "data: ")
	require.Contains(t, body, "\n\n")

	payload := body[len("data: "):]
	require.Equal(t, "chat.completion.chunk", gjson.Get(payload, "object").String())
	require.Equal(t, "gpt-5", gjson.Get(payload, "model").String())
	require.Equal(t, "assistant", gjson.Get(payload, "choices.0.delta.role").String())
	require.Equal(t, "", gjson.Get(payload, "choices.0.delta.content").String())
	require.False(t, gjson.Get(payload, "choices.0.finish_reason").Exists() &&
		gjson.Get(payload, "choices.0.finish_reason").String() != "")

	require.True(t, OpenAIChatFirstFrameInjected(c))
	require.True(t, len(OpenAIChatFirstFrameID(c)) > len("chatcmpl-"))

	// 幂等：同一请求不会重复下发。
	written := rec.Body.Len()
	require.False(t, WriteOpenAIChatFirstFrame(c, "gpt-5"))
	require.Equal(t, written, rec.Body.Len())
}

// 首帧字节必须从"是否已写出语义响应"的口径里扣除，否则上游 429/5xx 会因为
// 首帧已经发出而不再换号重试。
func TestOpenAIAdjustedWrittenSizeExcludesFirstFrame(t *testing.T) {
	c, _ := newFirstFrameTestContext(t)

	before := OpenAICompactKeepaliveAdjustedWrittenSize(c)
	require.True(t, WriteOpenAIChatFirstFrame(c, "gpt-5"))
	require.Equal(t, before, OpenAICompactKeepaliveAdjustedWrittenSize(c),
		"仅有首帧字节时口径必须不变，否则 failover 换号会被误判放弃")

	// 真实语义字节写出后口径必须变化。
	_, err := c.Writer.Write([]byte("data: real\n\n"))
	require.NoError(t, err)
	require.Equal(t, len("data: real\n\n"), OpenAICompactKeepaliveAdjustedWrittenSize(c))
}

// 首帧快返由运行时设置控制：key 缺失或显式关闭时必须保持关闭；settingService
// 未注入（单测/降级路径）也必须保持关闭。
func TestOpenAIChatFirstFrameEnabled_RuntimeSetting(t *testing.T) {
	ctx := context.Background()
	repo := &gatewayTTLSettingRepo{data: map[string]string{}}
	gatewayForwardingCache.Store(&cachedGatewayForwardingSettings{})
	svc := &OpenAIGatewayService{settingService: NewSettingService(repo, &config.Config{})}

	require.False(t, svc.ChatFirstFrameEnabled(ctx), "key 缺失时必须默认关闭")

	repo.data[SettingKeyOpenAIChatFirstFrameEnabled] = "false"
	gatewayForwardingCache.Store(&cachedGatewayForwardingSettings{})
	require.False(t, svc.ChatFirstFrameEnabled(ctx))

	repo.data[SettingKeyOpenAIChatFirstFrameEnabled] = "true"
	gatewayForwardingCache.Store(&cachedGatewayForwardingSettings{})
	require.True(t, svc.ChatFirstFrameEnabled(ctx))

	require.False(t, (&OpenAIGatewayService{}).ChatFirstFrameEnabled(ctx))
}

func TestStripInjectedChatRoleFromSSELine(t *testing.T) {
	stripped := false

	// 第一个带 role 的 chunk：删掉 role，保留 content。
	out := stripInjectedChatRoleFromSSELine(
		`data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"hi"}}]}`, &stripped)
	require.True(t, stripped)
	require.False(t, gjson.Get(out, "choices.0.delta.role").Exists())
	require.Equal(t, "hi", gjson.Get(out, "choices.0.delta.content").String())
	require.NotContains(t, out, "\n", "改写结果必须是单行，行尾换行由写出侧统一补")

	// 之后的 chunk 不再改动。
	second := `data: {"choices":[{"index":0,"delta":{"content":"more"}}]}`
	require.Equal(t, second, stripInjectedChatRoleFromSSELine(second, &stripped))

	// 非数据行与终止行原样透传。
	require.Equal(t, "", stripInjectedChatRoleFromSSELine("", &stripped))
	require.Equal(t, "data: [DONE]", stripInjectedChatRoleFromSSELine("data: [DONE]", &stripped))
	require.Equal(t, "event: ping", stripInjectedChatRoleFromSSELine("event: ping", &stripped))
}
