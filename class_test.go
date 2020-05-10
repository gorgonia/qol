package qol

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/suite"
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
	// chewxy testcase
	matF64 = T.New(T.WithBacking([]float64{0.1, 0.1, 0.6, 0.7, 0.1, 0.6, 0.1, 0.1, 0.7, 0.1}), T.WithShape(2, 5))
	assert.Equal(t, ToClasses(matF64, 0), []Class{2, 0})

	matF64 = T.New(T.WithBacking([]float64{0, 0, 1, 0, 1, 0}), shp)
	assert.Equal(t, ToClasses(matF64, 0), []Class{2, 1})

	matF64 = T.New(T.WithBacking([]float64{0.1, 0.6, 0.7, 0.1, 0.7, 0.6}), shp)
	assert.Equal(t, ToClasses(matF64, 0), []Class{1, 1})

	matF64 = T.New(T.WithBacking([]float64{0.1, 0.6, 0.7, 0.1, 0.7, 0.6}), shp)
	assert.Equal(t, ToClasses(matF64, 0.65), []Class{2, 1})

	matF64 = T.New(T.WithBacking([]float64{0.1, -1, 0.7, -1.0, 0.2, 0.6}), shp)
	assert.Equal(t, ToClasses(matF64, 0), []Class{2, 2})

	matF64 = T.New(T.WithBacking([]float64{1, 1, 1, 1, 1, 1}), shp)
	assert.Equal(t, ToClasses(matF64, 0), []Class{0, 0})

	// Int
	matInt = T.New(T.WithBacking([]int{0, 0, 1, 0, 1, 0}), shp)
	assert.Equal(t, ToClasses(matInt, 0), []Class{2, 1})

	matInt = T.New(T.WithBacking([]int{0, 0, 1, 0, 1, 0}), shp)
	assert.Equal(t, ToClasses(matInt, 999), []Class{2, 1})

	matInt = T.New(T.WithBacking([]int{0, 0, 2, 1, 2, 0}), shp)
	assert.Equal(t, ToClasses(matInt, 0), []Class{2, 0})

	matInt = T.New(T.WithBacking([]int{1, 1, 1, 1, 1, 1}), shp)
	assert.Equal(t, ToClasses(matInt, 0), []Class{0, 0})

	matInt = T.New(T.WithBacking([]int{-1, 1, -2, -3, 0, 1}), shp)
	assert.Equal(t, ToClasses(matInt, 0), []Class{1, 2})

	// Uint
	matUint = T.New(T.WithBacking([]uint{0, 0, 1, 0, 1, 0}), shp)
	assert.Equal(t, ToClasses(matUint, 0), []Class{2, 1})

	matUint = T.New(T.WithBacking([]uint{0, 0, 1, 0, 1, 0}), shp)
	assert.Equal(t, ToClasses(matUint, 999), []Class{2, 1})

	matUint = T.New(T.WithBacking([]uint{0, 0, 2, 1, 2, 0}), shp)
	assert.Equal(t, ToClasses(matUint, 0), []Class{2, 0})

	matUint = T.New(T.WithBacking([]uint{1, 1, 1, 1, 1, 1}), shp)
	assert.Equal(t, ToClasses(matUint, 0), []Class{0, 0})
}
func NewToOneHotVectorSuite(unsafe bool, a Class, numClasses uint, backingActual, backingExpected interface{}) *ToOneHotVectorSuite {
	shp := T.WithShape(int(numClasses))
	return &ToOneHotVectorSuite{
		unsafe:     unsafe,
		a:          a,
		numClasses: numClasses,
		reuse:      T.New(T.WithBacking(backingActual), shp),
		expected:   T.New(T.WithBacking(backingExpected), shp),
	}
}

// ToOneHotVectorSuite test both the safe and unsafe version and the
// ToOneHotVector by specifying the `unsafe` boolean flag
type ToOneHotVectorSuite struct {
	suite.Suite
	unsafe          bool
	a               Class
	numClasses      uint
	reuse, expected *T.Dense
}

func (suite *ToOneHotVectorSuite) Test() {
	// Safe or unsafe function
	var oh *T.Dense
	if suite.unsafe {
		oh = UnsafeToOneHotVector(suite.a, suite.numClasses, suite.reuse)
	} else {
		oh = ToOneHotVector(suite.a, suite.numClasses, suite.reuse.Dtype())
	}
	// Check data and shape between expected and resulting of UnsafeToOneHotVector
	assert.Equal(suite.T(), oh.Data(), suite.expected.Data())
	assert.Equal(suite.T(), oh.Shape(), suite.expected.Shape())
	if suite.unsafe {
		// Check if the operation is infact unsafe
		assert.Equal(suite.T(), suite.reuse.Data(), suite.expected.Data())
		assert.Equal(suite.T(), suite.reuse.Shape(), suite.expected.Shape())
		assert.Equal(suite.T(), &oh, &suite.reuse)
	}
}

func TestToOneHotVector(t *testing.T) {
	// Panics
	// n classes not the same as vector length
	assert.Panics(t, func() { UnsafeToOneHotVector(0, 999, T.New(T.Of(T.Float32), T.WithShape(5))) })
	assert.Panics(t, func() { UnsafeToOneHotVector(0, 2, T.New(T.Of(T.Float32), T.WithShape(5))) })
	assert.NotPanics(t, func() { UnsafeToOneHotVector(0, 5, T.New(T.Of(T.Float32), T.WithShape(5))) })
	// Class is out of range
	assert.Panics(t, func() { UnsafeToOneHotVector(10, 5, T.New(T.Of(T.Float32), T.WithShape(5))) })
	assert.Panics(t, func() { UnsafeToOneHotVector(5, 5, T.New(T.Of(T.Float32), T.WithShape(5))) })
	assert.NotPanics(t, func() { UnsafeToOneHotVector(0, 5, T.New(T.Of(T.Float32), T.WithShape(5))) })
	// Non Vector
	assert.Panics(t, func() { UnsafeToOneHotVector(0, 5, T.New(T.Of(T.Float32), T.WithShape(5, 5))) })
	assert.Panics(t, func() { UnsafeToOneHotVector(0, 5, T.New(T.Of(T.Float32), T.WithShape(1, 5))) })
	// Unsupported type
	assert.Panics(t, func() { UnsafeToOneHotVector(0, 5, T.New(T.Of(T.Complex64), T.WithShape(5))) })

	// Value tests
	// Float32
	suite.Run(t, NewToOneHotVectorSuite(false, 1, 5, []float32{0, 0, 0, 0, 0}, []float32{0, 1, 0, 0, 0}))
	suite.Run(t, NewToOneHotVectorSuite(false, 1, 5, []float32{1, 1, 1, 1, 1}, []float32{0, 1, 0, 0, 0}))
	suite.Run(t, NewToOneHotVectorSuite(false, 1, 5, []float32{1, 1, 1, 1, 1}, []float32{0, 1, 0, 0, 0}))
	suite.Run(t, NewToOneHotVectorSuite(false, 1, 3, []float32{0, 0, 0}, []float32{0, 1, 0}))
	suite.Run(t, NewToOneHotVectorSuite(false, 3, 5, []float32{0, 0, 0, 0, 0}, []float32{0, 0, 0, 1, 0}))
	// Float64
	suite.Run(t, NewToOneHotVectorSuite(false, 1, 5, []float64{0, 0, 0, 0, 0}, []float64{0, 1, 0, 0, 0}))
	suite.Run(t, NewToOneHotVectorSuite(false, 1, 5, []float64{1, 1, 1, 1, 1}, []float64{0, 1, 0, 0, 0}))
	suite.Run(t, NewToOneHotVectorSuite(false, 1, 5, []float64{1, 1, 1, 1, 1}, []float64{0, 1, 0, 0, 0}))
	suite.Run(t, NewToOneHotVectorSuite(false, 1, 3, []float64{0, 0, 0}, []float64{0, 1, 0}))
	suite.Run(t, NewToOneHotVectorSuite(false, 3, 5, []float64{0, 0, 0, 0, 0}, []float64{0, 0, 0, 1, 0}))
	// Int32
	suite.Run(t, NewToOneHotVectorSuite(false, 1, 5, []int32{0, 0, 0, 0, 0}, []int32{0, 1, 0, 0, 0}))
	suite.Run(t, NewToOneHotVectorSuite(false, 1, 5, []int32{1, 1, 1, 1, 1}, []int32{0, 1, 0, 0, 0}))
	suite.Run(t, NewToOneHotVectorSuite(false, 1, 5, []int32{1, 1, 1, 1, 1}, []int32{0, 1, 0, 0, 0}))
	suite.Run(t, NewToOneHotVectorSuite(false, 1, 3, []int32{0, 0, 0}, []int32{0, 1, 0}))
	suite.Run(t, NewToOneHotVectorSuite(false, 3, 5, []int32{0, 0, 0, 0, 0}, []int32{0, 0, 0, 1, 0}))
	// Int64
	suite.Run(t, NewToOneHotVectorSuite(false, 1, 5, []int64{0, 0, 0, 0, 0}, []int64{0, 1, 0, 0, 0}))
	suite.Run(t, NewToOneHotVectorSuite(false, 1, 5, []int64{1, 1, 1, 1, 1}, []int64{0, 1, 0, 0, 0}))
	suite.Run(t, NewToOneHotVectorSuite(false, 1, 5, []int64{1, 1, 1, 1, 1}, []int64{0, 1, 0, 0, 0}))
	suite.Run(t, NewToOneHotVectorSuite(false, 1, 3, []int64{0, 0, 0}, []int64{0, 1, 0}))
	suite.Run(t, NewToOneHotVectorSuite(false, 3, 5, []int64{0, 0, 0, 0, 0}, []int64{0, 0, 0, 1, 0}))
	// Int
	suite.Run(t, NewToOneHotVectorSuite(false, 1, 5, []int{0, 0, 0, 0, 0}, []int{0, 1, 0, 0, 0}))
	suite.Run(t, NewToOneHotVectorSuite(false, 1, 5, []int{1, 1, 1, 1, 1}, []int{0, 1, 0, 0, 0}))
	suite.Run(t, NewToOneHotVectorSuite(false, 1, 5, []int{1, 1, 1, 1, 1}, []int{0, 1, 0, 0, 0}))
	suite.Run(t, NewToOneHotVectorSuite(false, 1, 3, []int{0, 0, 0}, []int{0, 1, 0}))
	suite.Run(t, NewToOneHotVectorSuite(false, 3, 5, []int{0, 0, 0, 0, 0}, []int{0, 0, 0, 1, 0}))
}

func TestToOneHotVectorSuite(t *testing.T) {
	// Panics
	// n classes not the same as vector length
	assert.Panics(t, func() { UnsafeToOneHotVector(0, 999, T.New(T.Of(T.Float32), T.WithShape(5))) })
	assert.Panics(t, func() { UnsafeToOneHotVector(0, 2, T.New(T.Of(T.Float32), T.WithShape(5))) })
	assert.NotPanics(t, func() { UnsafeToOneHotVector(0, 5, T.New(T.Of(T.Float32), T.WithShape(5))) })
	// Class is out of range
	assert.Panics(t, func() { UnsafeToOneHotVector(10, 5, T.New(T.Of(T.Float32), T.WithShape(5))) })
	assert.Panics(t, func() { UnsafeToOneHotVector(5, 5, T.New(T.Of(T.Float32), T.WithShape(5))) })
	assert.NotPanics(t, func() { UnsafeToOneHotVector(0, 5, T.New(T.Of(T.Float32), T.WithShape(5))) })
	// Non Vector
	assert.Panics(t, func() { UnsafeToOneHotVector(0, 5, T.New(T.Of(T.Float32), T.WithShape(5, 5))) })
	assert.Panics(t, func() { UnsafeToOneHotVector(0, 5, T.New(T.Of(T.Float32), T.WithShape(1, 5))) })
	// Unsupported type
	assert.Panics(t, func() { UnsafeToOneHotVector(0, 5, T.New(T.Of(T.Complex64), T.WithShape(5))) })

	// Value tests
	// Float32
	suite.Run(t, NewToOneHotVectorSuite(true, 1, 5, []float32{0, 0, 0, 0, 0}, []float32{0, 1, 0, 0, 0}))
	suite.Run(t, NewToOneHotVectorSuite(true, 1, 5, []float32{1, 1, 1, 1, 1}, []float32{0, 1, 0, 0, 0}))
	suite.Run(t, NewToOneHotVectorSuite(true, 1, 5, []float32{1, 1, 1, 1, 1}, []float32{0, 1, 0, 0, 0}))
	suite.Run(t, NewToOneHotVectorSuite(true, 1, 3, []float32{0, 0, 0}, []float32{0, 1, 0}))
	suite.Run(t, NewToOneHotVectorSuite(true, 3, 5, []float32{0, 0, 0, 0, 0}, []float32{0, 0, 0, 1, 0}))
	// Float64
	suite.Run(t, NewToOneHotVectorSuite(true, 1, 5, []float64{0, 0, 0, 0, 0}, []float64{0, 1, 0, 0, 0}))
	suite.Run(t, NewToOneHotVectorSuite(true, 1, 5, []float64{1, 1, 1, 1, 1}, []float64{0, 1, 0, 0, 0}))
	suite.Run(t, NewToOneHotVectorSuite(true, 1, 5, []float64{1, 1, 1, 1, 1}, []float64{0, 1, 0, 0, 0}))
	suite.Run(t, NewToOneHotVectorSuite(true, 1, 3, []float64{0, 0, 0}, []float64{0, 1, 0}))
	suite.Run(t, NewToOneHotVectorSuite(true, 3, 5, []float64{0, 0, 0, 0, 0}, []float64{0, 0, 0, 1, 0}))
	// Int32
	suite.Run(t, NewToOneHotVectorSuite(true, 1, 5, []int32{0, 0, 0, 0, 0}, []int32{0, 1, 0, 0, 0}))
	suite.Run(t, NewToOneHotVectorSuite(true, 1, 5, []int32{1, 1, 1, 1, 1}, []int32{0, 1, 0, 0, 0}))
	suite.Run(t, NewToOneHotVectorSuite(true, 1, 5, []int32{1, 1, 1, 1, 1}, []int32{0, 1, 0, 0, 0}))
	suite.Run(t, NewToOneHotVectorSuite(true, 1, 3, []int32{0, 0, 0}, []int32{0, 1, 0}))
	suite.Run(t, NewToOneHotVectorSuite(true, 3, 5, []int32{0, 0, 0, 0, 0}, []int32{0, 0, 0, 1, 0}))
	// Int64
	suite.Run(t, NewToOneHotVectorSuite(true, 1, 5, []int64{0, 0, 0, 0, 0}, []int64{0, 1, 0, 0, 0}))
	suite.Run(t, NewToOneHotVectorSuite(true, 1, 5, []int64{1, 1, 1, 1, 1}, []int64{0, 1, 0, 0, 0}))
	suite.Run(t, NewToOneHotVectorSuite(true, 1, 5, []int64{1, 1, 1, 1, 1}, []int64{0, 1, 0, 0, 0}))
	suite.Run(t, NewToOneHotVectorSuite(true, 1, 3, []int64{0, 0, 0}, []int64{0, 1, 0}))
	suite.Run(t, NewToOneHotVectorSuite(true, 3, 5, []int64{0, 0, 0, 0, 0}, []int64{0, 0, 0, 1, 0}))
	// Int
	suite.Run(t, NewToOneHotVectorSuite(true, 1, 5, []int{0, 0, 0, 0, 0}, []int{0, 1, 0, 0, 0}))
	suite.Run(t, NewToOneHotVectorSuite(true, 1, 5, []int{1, 1, 1, 1, 1}, []int{0, 1, 0, 0, 0}))
	suite.Run(t, NewToOneHotVectorSuite(true, 1, 5, []int{1, 1, 1, 1, 1}, []int{0, 1, 0, 0, 0}))
	suite.Run(t, NewToOneHotVectorSuite(true, 1, 3, []int{0, 0, 0}, []int{0, 1, 0}))
	suite.Run(t, NewToOneHotVectorSuite(true, 3, 5, []int{0, 0, 0, 0, 0}, []int{0, 0, 0, 1, 0}))
}

func TestToOneHotMatrix(t *testing.T) {
	// Not Implemented
}

func TestUnsafeToOneHotMatrix(t *testing.T) {

	// Not Implemented
}
