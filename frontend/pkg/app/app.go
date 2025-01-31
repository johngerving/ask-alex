package app

import (
	"errors"
	"log/slog"
	"net/http"
	"os"

	"github.com/r3labs/sse/v2"
)

// Struct for the main app
type App struct {
	config    Config
	logger    *slog.Logger
	sseServer *sse.Server
}

// New() creates a new *App and returns it.
func New() (*App, error) {
	config := Config{}

	// Set up a logger
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))

	// Create a server for SSE streams
	sseServer := sse.New()

	return &App{
		config:    config,
		logger:    logger,
		sseServer: sseServer,
	}, nil
}

// Start() registers handlers with routes and starts an HTTP server.
func (a *App) Start() error {
	// Register routes
	e, err := a.registerRoutes()
	if err != nil {
		return err
	}

	// Start the server
	if err := e.Start(":8080"); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}

	return nil
}
