// Copyright 2026 The Bee Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
	"embed"
	"fmt"
	"io"
)

//go:embed books/*
var Books embed.FS

// Book is a book
type Book struct {
	Name  string
	Text  []byte
	Real  bool
	Index int
}

// LoadBooks loads books
func LoadBooks() []Book {
	books := []Book{
		{
			Real:  true,
			Name:  "10.txt.utf-8.bz2",
			Index: 0,
		},
	}
	load := func(book string) []byte {
		file, err := Books.Open(book)
		if err != nil {
			panic(err)
		}
		defer file.Close()
		breader := bzip2.NewReader(file)
		data, err := io.ReadAll(breader)
		if err != nil {
			panic(err)
		}
		return data
	}
	for i := range books {
		books[i].Text = load(fmt.Sprintf("books/%s", books[i].Name))
	}
	return books
}

func main() {
	books := LoadBooks()
	type Context struct {
		Distribution [256]uint8
		Probability  uint8
	}
	var model [256]Context
	text := books[0].Text
	var context uint8
	for _, symbol := range text {
		if model[context].Distribution[symbol] > 33 {
			for i := range model[context].Distribution {
				model[context].Distribution[i] >>= 1
			}
		}
		model[context].Distribution[symbol]++
		context = symbol
	}
}
