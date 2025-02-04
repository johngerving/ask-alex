package handler

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"

	"github.com/gomarkdown/markdown"
	"github.com/google/uuid"
	"github.com/johngerving/ask-alex.git/pkg/templates"
	"github.com/johngerving/ask-alex.git/pkg/views"
	"github.com/labstack/echo/v4"
	"github.com/microcosm-cc/bluemonday"
	"github.com/r3labs/sse/v2"
)

// GET /chat
func ChatPageGET() echo.HandlerFunc {
	return func(c echo.Context) error {
		return views.Chat().Render(context.Background(), c.Response().Writer)
	}
}

// GET /chat/responses
// Streams an LLM response to the client - listens on /chat/responses?stream=<STREAM_ID>.
// The client will then receive events from the stream STREAM_ID.
func LLMResponseGET(l *slog.Logger, sseServer *sse.Server) echo.HandlerFunc {
	return func(c echo.Context) error {
		streamID := c.Request().URL.Query().Get("stream")
		if streamID == "" {
			http.Error(c.Response().Writer, "Please specify a stream!", http.StatusInternalServerError)
			return fmt.Errorf("stream not specified")
		}
		// TODO: Add user authentication to message route
		go func() {
			<-c.Request().Context().Done() // Received Browser Disconnection
			l.Info("Client disconnected")
			sseServer.RemoveStream(streamID)
		}()

		sseServer.ServeHTTP(c.Response().Writer, c.Request())
		return nil
	}
}

// POST /chat/messages
// Create a new LLM response and publish it to an SSE stream.
func ChatMessagePOST(l *slog.Logger, sseServer *sse.Server) echo.HandlerFunc {
	return func(c echo.Context) error {
		vals, err := c.FormParams()

		// Get the user message
		if err != nil || vals.Get("message") == "" {
			return fmt.Errorf("invalid form parameters")
		}

		// TODO: authenticate user

		// Create a unique stream ID
		streamID := fmt.Sprintf("stream-%v", uuid.New())
		streamURL := fmt.Sprintf("/chat/responses?stream=%v", streamID)

		// Create a stream to publish response updates to
		sseServer.CreateStream(streamID)

		// Initial components to return to the user
		chatBubbleComponent := templates.ChatBubble(vals.Get("message"), true, false, "")
		llmResponseComponent := templates.ChatBubble("", false, true, streamURL)
		chatFormComponent := templates.ChatForm()

		chatBubbleComponent.Render(context.Background(), c.Response().Writer)
		llmResponseComponent.Render(context.Background(), c.Response().Writer)
		chatFormComponent.Render(context.Background(), c.Response().Writer)

		go func(s *sse.Server) {
			// defer sseServer.RemoveStream(streamID)
			defer s.Publish(streamID, &sse.Event{
				Event: []byte("done"),
				Data:  []byte("test"),
			})

			reqBody := strings.NewReader(vals.Get("message"))

			// Make a request to the RAG endpoint with the query
			res, err := http.DefaultClient.Post("http://localhost:8081", "text/plain", reqBody)
			if err != nil {
				l.Error(err.Error())
				return
			}
			defer res.Body.Close()

			if res.StatusCode != http.StatusOK {
				l.Error(fmt.Sprintf("RAG response status code: %v", res.StatusCode))
				return
			}

			// Get the response from the RAG pipeline
			bodyBytes, err := io.ReadAll(res.Body)
			if err != nil {
				l.Error(err.Error())
				return
			}

			// Convert markdown from LLM to HTML
			llmResponseHTML := bluemonday.UGCPolicy().SanitizeBytes(markdown.ToHTML(bodyBytes, nil, nil))
			llmResponseHTML = []byte(strings.Replace(string(llmResponseHTML), "\n", "<br>", -1))

			l.Info(streamID)
			// Publish the response to the SSE stream
			s.Publish(streamID, &sse.Event{
				Event: []byte("message"),
				Data:  llmResponseHTML,
			})

			l.Info(streamID)
			// Publish another event saying that we're done - this time, send the entire chat bubble component so that the
			// client doesn't keep listening on the stream
			l.Info("done")
		}(sseServer)

		return nil
	}
}
