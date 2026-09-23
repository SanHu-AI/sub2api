package handler

import (
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/stretchr/testify/require"
)

// 首帧快返默认关闭：它会把响应头提前固化为 200 + SSE，只应由部署方显式开启。
func TestOpenAIChatFirstFrameEnabledDefaultsOff(t *testing.T) {
	h := &OpenAIGatewayHandler{}
	require.False(t, h.openAIChatFirstFrameEnabled())

	h.cfg = &config.Config{}
	require.False(t, h.openAIChatFirstFrameEnabled())

	h.cfg.Gateway.ChatFirstFrameEnabled = true
	require.True(t, h.openAIChatFirstFrameEnabled())
}
