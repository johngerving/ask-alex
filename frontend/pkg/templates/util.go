package templates

import (
	"bytes"
	"context"

	"github.com/a-h/templ"
)

// TemplateToBytes converts a templ component to a byte array.
func TemplateToBytes(t templ.Component) ([]byte, error) {
	var b bytes.Buffer
	if err := t.Render(context.Background(), &b); err != nil {
		return []byte{}, err
	}
	return b.Bytes(), nil
}
