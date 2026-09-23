package apicompat

import (
	"testing"

	"github.com/stretchr/testify/require"
)

// 首帧快返会预置 SentRole：上游 response.created 到达时不能再下发一次
// assistant 角色帧，但仍必须把上游的 ID / Model / ServiceTier 刷进 state，
// 否则整条流会一直使用占位 ID。
func TestResponsesEventToChatChunks_CreatedDoesNotRepeatRoleButRefreshesState(t *testing.T) {
	state := NewResponsesEventToChatState()
	state.SentRole = true
	state.ID = "chatcmpl-placeholder"
	state.Model = ""

	evt := &ResponsesStreamEvent{
		Type: "response.created",
		Response: &ResponsesResponse{
			ID:          "resp_upstream_1",
			Model:       "gpt-5",
			ServiceTier: "priority",
		},
	}

	chunks := ResponsesEventToChatChunks(evt, state)
	require.Empty(t, chunks, "已下发过 role 时不得重复下发角色帧")
	require.Equal(t, "resp_upstream_1", state.ID)
	require.Equal(t, "gpt-5", state.Model)
	require.Equal(t, "priority", state.ServiceTier)
}

// 未预置 SentRole 时行为保持原样：下发一次角色帧。
func TestResponsesEventToChatChunks_CreatedEmitsRoleOnce(t *testing.T) {
	state := NewResponsesEventToChatState()
	evt := &ResponsesStreamEvent{
		Type:     "response.created",
		Response: &ResponsesResponse{ID: "resp_upstream_1", Model: "gpt-5"},
	}

	chunks := ResponsesEventToChatChunks(evt, state)
	require.Len(t, chunks, 1)
	require.Equal(t, "assistant", chunks[0].Choices[0].Delta.Role)
	require.True(t, state.SentRole)

	again := ResponsesEventToChatChunks(evt, state)
	require.Empty(t, again)
}
