package qol

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
	T "gorgonia.org/tensor"
)

func TestToClass(t *testing.T) {
	// Panic Tests
	// TODO: panic without an iterator
	// Test panic on non vector
	assert.Panics(t, func() { ToClass(T.New(T.Of(T.Float32), T.WithShape(10, 10)), 0) })
	// Panic on unsupported T dtype
	assert.Panics(t, func() { ToClass(T.New(T.Of(T.Complex64), T.WithShape(10)), 0) })
	// Panic on Unreachable
	assert.Panics(t, func() { ToClass(T.New(T.WithBacking([]float32{0, 0, 1, 0, 0})), 999) })
	// Value Tests
	// only specifying vectors of length 5 varying values, type, and threshold
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
	// Panic Tests
	// TODO: panic without an iterator
	// Test panic on non matrix
	assert.Panics(t, func() { ToClasses(T.New(T.Of(T.Float32), T.WithShape(10)), 0) })
	assert.Panics(t, func() { ToClasses(T.New(T.Of(T.Float32), T.WithShape(10, 10, 10)), 0) })
	// Panic on unsupported T dtype
	assert.Panics(t, func() { ToClasses(T.New(T.Of(T.Complex64), T.WithShape(10)), 0) })
	// Panic on Unreachable
	assert.Panics(t, func() { ToClasses(T.New(T.WithBacking([]float32{0, 0, 1, 0, 0, 1}), T.WithShape(2, 3)), 999) })
	// Value Tests
	// only specifying matracies of length 2x3 varying values, type, and threshold
	var matF32, matF64, matInt, matUint tensor.Tensor
	shp := T.WithShape(2, 3)
	// Float32
	// chewxy testcase
	matF32 = T.New(T.WithBacking([]float32{0.1, 0.1, 0.6, 0.7, 0.1, 0.6, 0.1, 0.1, 0.7, 0.1}), T.WithShape(2, 5))
	assert.Equal(t, ToClasses(matF32, 0), []Class{2, 0})

	matF32 = T.New(T.WithBacking([]float32{0, 0, 1, 0, 1, 0}), shp)
	assert.Equal(t, ToClasses(matF32, 0), []Class{2, 1})

	matF32 = T.New(T.WithBacking([]float32{0.1, 0.6, 0.7, 0.1, 0.7, 0.6}), shp)
	assert.Equal(t, ToClasses(matF32, 0), []Class{1, 1})

	matF32 = T.New(T.WithBacking([]float32{0.1, 0.6, 0.7, 0.1, 0.7, 0.6}), shp)
	assert.Equal(t, ToClasses(matF32, 0.65), []Class{2, 1})

	matF32 = T.New(T.WithBacking([]float32{0.1, -1, 0.7, -1.0, 0.2, 0.6}), shp)
	assert.Equal(t, ToClasses(matF32, 0), []Class{2, 2})

	matF32 = T.New(T.WithBacking([]float32{1, 1, 1, 1, 1, 1}), shp)
	assert.Equal(t, ToClasses(matF32, 0), []Class{0, 0})

	// Float64
	matF64 = T.New(T.WithBacking([]float64{0, 0, 1, 0, 1, 0}), shp)
	assert.Equal(t, ToClasses(matF64, 0), []Class{2, 1})

	// Int
	matInt = T.New(T.WithBacking([]int{0, 0, 1, 0, 1, 0}), shp)
	assert.Equal(t, ToClasses(matInt, 0), []Class{2, 1})
	// Uint
	matUint = T.New(T.WithBacking([]uint{0, 0, 1, 0, 1, 0}), shp)
	assert.Equal(t, ToClasses(matUint, 0), []Class{2, 1})
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
