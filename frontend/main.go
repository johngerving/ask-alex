package main

import (
	"log"

	"github.com/johngerving/ask-alex.git/pkg/app"
)

func main() {
	a, err := app.New()
	if err != nil {
		log.Fatal(err)
	}

	if err := a.Start(); err != nil {
		log.Fatal(err)
	}
}
