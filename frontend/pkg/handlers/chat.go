package handlers

import (
	"bytes"
	"context"
	"fmt"
	"io"

	"github.com/johngerving/ask-alex.git/pkg/templates"
	"github.com/johngerving/ask-alex.git/pkg/views"
	"github.com/labstack/echo/v4"
)

// Event represents Server-Sent Event.
// SSE explanation: https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format
type Event struct {
	// ID is used to set the EventSource object's last event ID value.
	ID []byte
	// Data field is for the message. When the EventSource receives multiple consecutive lines
	// that begin with data:, it concatenates them, inserting a newline character between each one.
	// Trailing newlines are removed.
	Data []byte
	// Event is a string identifying the type of event described. If this is specified, an event
	// will be dispatched on the browser to the listener for the specified event name; the website
	// source code should use addEventListener() to listen for named events. The onmessage handler
	// is called if no event name is specified for a message.
	Event []byte
	// Retry is the reconnection time. If the connection to the server is lost, the browser will
	// wait for the specified time before attempting to reconnect. This must be an integer, specifying
	// the reconnection time in milliseconds. If a non-integer value is specified, the field is ignored.
	Retry []byte
	// Comment line can be used to prevent connections from timing out; a server can send a comment
	// periodically to keep the connection alive.
	Comment []byte
}

// MarshalTo marshals Event to given Writer
func (ev *Event) MarshalTo(w io.Writer) error {
	// Marshalling part is taken from: https://github.com/r3labs/sse/blob/c6d5381ee3ca63828b321c16baa008fd6c0b4564/http.go#L16
	if len(ev.Data) == 0 && len(ev.Comment) == 0 {
		return nil
	}

	if len(ev.Data) > 0 {
		if _, err := fmt.Fprintf(w, "id: %s\n", ev.ID); err != nil {
			return err
		}

		sd := bytes.Split(ev.Data, []byte("\n"))
		for i := range sd {
			if _, err := fmt.Fprintf(w, "data: %s\n", sd[i]); err != nil {
				return err
			}
		}

		if len(ev.Event) > 0 {
			if _, err := fmt.Fprintf(w, "event: %s\n", ev.Event); err != nil {
				return err
			}
		}

		if len(ev.Retry) > 0 {
			if _, err := fmt.Fprintf(w, "retry: %s\n", ev.Retry); err != nil {
				return err
			}
		}
	}

	if len(ev.Comment) > 0 {
		if _, err := fmt.Fprintf(w, ": %s\n", ev.Comment); err != nil {
			return err
		}
	}

	if _, err := fmt.Fprint(w, "\n"); err != nil {
		return err
	}

	return nil
}

// GET /chat
func ChatPageHandler(c echo.Context) error {
	return views.Chat().Render(context.Background(), c.Response().Writer)
}

// GET /chat/messages
// Sends server-sent events to the client with any new
// messages
func ChatMessageGETHandler(c echo.Context) error {
	l := c.Echo().Logger
	l.Info("SSE client connected, ip: %v", c.RealIP())

	w := c.Response()
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	return nil
	// ticker := time.NewTicker(1 * time.Second)
	// defer ticker.Stop()
	// for {
	// 	select {
	// 	case <-c.Request().Context().Done():
	// 		l.Info("SSE client disconnected, ip: %v", c.RealIP())
	// 		return nil
	// 	case <-ticker.C:
	// 		component, err := templates.TemplateToBytes(templates.ChatBubble())
	// 		if err != nil {
	// 			return err
	// 		}

	// 		event := Event{
	// 			Data: component,
	// 		}
	// 		if err := event.MarshalTo(w); err != nil {
	// 			return err
	// 		}
	// 		w.Flush()
	// 	}
	// }
}

func ChatMessagePOSTHandler(c echo.Context) error {
	l := c.Logger()

	vals, err := c.FormParams()

	if err != nil {
		l.Errorf("Could not get form parameters: %v", err)
		return fmt.Errorf("invalid form parameters")
	}
	if vals.Get("message") == "" {
		return nil
	}

	chatBubbleComponent := templates.ChatBubble(vals.Get("message"), true)
	err = chatBubbleComponent.Render(context.Background(), c.Response().Writer)
	if err != nil {
		return err
	}

	chatFormComponent := templates.ChatForm()
	return chatFormComponent.Render(context.Background(), c.Response().Writer)
}
