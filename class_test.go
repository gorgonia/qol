package qol

import (
	"testing"

	"github.com/stretchr/testify/assert"
	T "gorgonia.org/tensor"
)

func TestToClass(t *testing.T) {
	// TODO: panic without an iterator
	// Test panic on non vector
	assert.Panics(t, func() { ToClass(T.New(T.Of(T.Float32), T.WithShape(10, 10)), 0) })
	// Panic on unsupported T dtype
	assert.Panics(t, func() { ToClass(T.New(T.Of(T.Complex64), T.WithShape(10)), 0) })
	// Panic on Unreachable
	assert.Panics(t, func() { ToClass(T.New(T.WithBacking([]float32{0, 0, 1, 0, 0})), 999) })

	// From here On only specifying vectors of length 5 varying type and threshold
	// Float32
	assert.Equal(t, ToClass(T.New(T.WithBacking([]float32{0, 0, 1, 0, 0})), 0), Class(2))
	assert.Equal(t, ToClass(T.New(T.WithBacking([]float32{0.1, 0.1, 0.6, 0.7, 0.1})), 0), Class(2))
	assert.Equal(t, ToClass(T.New(T.WithBacking([]float32{0.1, 0.1, 0.6, 0.7, 0.1})), 0.65), Class(3))
	assert.Equal(t, ToClass(T.New(T.WithBacking([]float32{1, 1, 1, 1, 1})), 0), Class(0))
	// Float64
	assert.Equal(t, ToClass(T.New(T.WithBacking([]float64{0, 0, 1, 0, 0})), 0), Class(2))
	assert.Equal(t, ToClass(T.New(T.WithBacking([]float64{0.1, 0.1, 0.6, 0.7, 0.1})), 0), Class(2))
	assert.Equal(t, ToClass(T.New(T.WithBacking([]float64{0.1, 0.1, 0.6, 0.7, 0.1})), 0.65), Class(3))
	assert.Equal(t, ToClass(T.New(T.WithBacking([]float64{1, 1, 1, 1, 1})), 0), Class(0))
	// Int
	assert.Equal(t, ToClass(T.New(T.WithBacking([]int{0, 0, 1, 0, 0})), 0), Class(2))
	assert.Equal(t, ToClass(T.New(T.WithBacking([]int{0, 0, 1, 0, 0})), 999), Class(2))
	assert.Equal(t, ToClass(T.New(T.WithBacking([]int{0, 0, 2, 0, 0})), 0), Class(2))
	assert.Equal(t, ToClass(T.New(T.WithBacking([]int{0, 1, 2, 0, 0})), 0), Class(1))
	assert.Equal(t, ToClass(T.New(T.WithBacking([]int{1, 1, 1, 1, 1})), 0), Class(0))
	assert.Equal(t, ToClass(T.New(T.WithBacking([]int{-1, -2, -3, 0, 1})), 0), Class(4))
	// uint
	assert.Equal(t, ToClass(T.New(T.WithBacking([]uint{0, 0, 1, 0, 0})), 0), Class(2))
	assert.Equal(t, ToClass(T.New(T.WithBacking([]uint{0, 0, 1, 0, 0})), 999), Class(2))
	assert.Equal(t, ToClass(T.New(T.WithBacking([]uint{0, 0, 2, 0, 0})), 0), Class(2))
	assert.Equal(t, ToClass(T.New(T.WithBacking([]uint{0, 1, 2, 0, 0})), 0), Class(1))
	assert.Equal(t, ToClass(T.New(T.WithBacking([]uint{1, 1, 1, 1, 1})), 0), Class(0))
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
