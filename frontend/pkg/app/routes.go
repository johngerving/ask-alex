package app

import (
	"context"
	"log/slog"

	"github.com/johngerving/ask-alex.git/pkg/handler"
	"github.com/labstack/echo/v4"
	"github.com/labstack/echo/v4/middleware"
)

// registerRoutes registers both page routes and other routes with an Echo router and returns the router.
func (a *App) registerRoutes() (*echo.Echo, error) {
	e := echo.New()

	// Serve static files
	e.Static("/static", "static")

	// Set up HTTP logger
	e.Use(middleware.RequestLoggerWithConfig(middleware.RequestLoggerConfig{
		LogStatus:   true,
		LogURI:      true,
		LogError:    true,
		HandleError: true, // forwards error to the global error handler, so it can decide appropriate status code
		LogValuesFunc: func(c echo.Context, v middleware.RequestLoggerValues) error {
			if v.Error == nil {
				a.logger.LogAttrs(context.Background(), slog.LevelInfo, "REQUEST",
					slog.String("uri", v.URI),
					slog.Int("status", v.Status),
				)
			} else {
				a.logger.LogAttrs(context.Background(), slog.LevelError, "REQUEST_ERROR",
					slog.String("uri", v.URI),
					slog.Int("status", v.Status),
					slog.String("err", v.Error.Error()),
				)
			}
			return nil
		},
	}))

	// Register page routes - these serve HTML
	err := a.registerPageRoutes(e)
	if err != nil {
		return nil, err
	}

	// Register all other routes
	e.GET("/chat/responses", handler.LLMResponseGET(a.logger, a.sseServer))
	e.POST("/chat/messages", handler.ChatMessagePOST(a.logger, a.sseServer))

	return e, nil
}

// registerPageRoutes registers routes for HTML pages
func (a *App) registerPageRoutes(e *echo.Echo) error {
	e.GET("/chat", handler.ChatPageGET())

	return nil
}
