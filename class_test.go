package qol

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestToClass(t *testing.T) {
	// TODO: panic without an iterator
	// Test panic on non vector
	panicOnNonVector := func() {
		_ = ToClass(tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(10, 10)), 0)
	}
	assert.Panics(t, panicOnNonVector)
	// Panic on unsupported tensor dtype
	panicOnUnsupportedType := func() {
		_ = ToClass(tensor.New(tensor.Of(tensor.Complex64), tensor.WithShape(10)), 0)
	}
	assert.Panics(t, panicOnUnsupportedType)

	// // Float32
	// dataFloat32 := []float32{0, 0, 1, 0, 0}
	// resultFloat32 := ToClass(tensor.New(tensor.WithBacking(dataFloat32)), 0)
	// assert.Equal(t, resultFloat32, Class(2))

	// dataFloat64 := []float64{0, 1, 2, 3, 4}
	// t.Log(ToClass(tensor.New(tensor.WithBacking(dataFloat64)), 0))

	// dataInt := []int{0, 1, 2, 3, 4}
	// t.Log(ToClass(tensor.New(tensor.WithBacking(dataInt)), 0))

	// dataUint := []uint{0, 1, 2, 3, 4}
	// t.Log(ToClass(tensor.New(tensor.WithBacking(dataUint)), 0))
}

func TestToClasses(t *testing.T) {
	// Not Implemented
}

func TestToOneHotVector(t *testing.T) {
	// Not Implemented
}

func TestToOneHotMatrix(t *testing.T) {

	// Not Implemented
}

func TestUnsafeToOneHotVector(t *testing.T) {

	// Not Implemented
}

func TestUnsafeToOneHotMatrix(t *testing.T) {

	// Not Implemented
}
