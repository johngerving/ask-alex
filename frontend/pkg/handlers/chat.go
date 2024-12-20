package handlers

import (
	"context"

	"github.com/johngerving/ask-alex.git/pkg/views"
	"github.com/labstack/echo/v4"
)

// GET /chat
func ChatHandler(c echo.Context) error {
	return views.Chat().Render(context.Background(), c.Response().Writer)
}
