// Copyright 2026 The Bee Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
	"embed"
	"fmt"
	"io"
	"math/rand"

	"github.com/pointlander/gradient"
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
	ctxt := gradient.Context[float64]{}
	set := ctxt.NewSet()
	set.Add("w0", 256, 256)
	set.AddBias("b0", 256)
	set.Add("w1", 512, 256)
	set.AddBias("b1", 256)
	set.AddData("input", 256)
	set.AddData("output", 256)
	rng := rand.New(rand.NewSource(1))
	set.InitAdam(rng)

	input := set.ByName["input"]
	input.X = input.X[:cap(input.X)]
	output := set.ByName["output"]
	output.X = output.X[:cap(output.X)]

	Add := ctxt.B(ctxt.Add)
	Mul := ctxt.B(ctxt.Mul)
	Everett := ctxt.U(ctxt.Everett)
	Sigmoid := ctxt.U(ctxt.Sigmoid)
	Quadratic := ctxt.B(ctxt.Quadratic)

	l0 := Everett(Add(Mul(set.Get("w0"), set.Get("input")), set.Get("b0")))
	l1 := Sigmoid(Add(Mul(set.Get("w1"), l0), set.Get("b1")))
	loss := Quadratic(l1, set.Get("output"))

	for i := range model {
		for ii := range model[i].Distribution {
			model[i].Distribution[ii] = 1
		}
	}

	for _, symbol := range text {
		for i := range model {
			model[i].Probability = 0
		}
		s := byte(rng.Intn(256))
		for range 32 * 256 {
			sum := 0
			for i := range model[s].Distribution {
				sum += int(model[s].Distribution[i])
			}
			selected, total := rng.Intn(sum), 0
			for i, value := range model[s].Distribution {
				total += int(value)
				if selected < total {
					if model[i].Probability > 33 {
						for ii := range model {
							model[ii].Probability >>= 1
						}
					}
					model[i].Probability++
					s = byte(i)
					break
				}
			}
		}
		sum := 0.0
		for i := range model {
			sum += float64(model[i].Probability)
		}
		for i := range model {
			input.X[i] = float64(model[i].Probability) / sum
		}
		for i := range model {
			if byte(i) == symbol {
				output.X[i] = 1.0
			} else {
				output.X[i] = 0.0
			}
		}

		set.Zero()
		l := gradient.Gradient(loss)
		set.Adam(gradient.B1, gradient.B2, .001)
		fmt.Println(l.X[0])

		if model[context].Distribution[symbol] > 33 {
			for i := range model[context].Distribution {
				model[context].Distribution[i] >>= 1
				if model[context].Distribution[i] == 0 {
					model[context].Distribution[i] = 1
				}
			}
		}
		model[context].Distribution[symbol]++
		context = symbol
	}
}
