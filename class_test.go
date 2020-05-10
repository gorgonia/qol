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
	assert.Equal(t, Class(2), ToClass(T.New(T.WithBacking([]float32{0, 0, 1, 0, 0})), 0))
	assert.Equal(t, Class(2), ToClass(T.New(T.WithBacking([]float32{0.1, 0.1, 0.6, 0.7, 0.1})), 0))
	assert.Equal(t, Class(3), ToClass(T.New(T.WithBacking([]float32{0.1, 0.1, 0.6, 0.7, 0.1})), 0.65))
	assert.Equal(t, Class(0), ToClass(T.New(T.WithBacking([]float32{1, 1, 1, 1, 1})), 0))
	// lFat64
	assert.Equal(t, Class(2), ToClass(T.New(T.WithBacking([]float64{0, 0, 1, 0, 0})), 0))
	assert.Equal(t, Class(2), ToClass(T.New(T.WithBacking([]float64{0.1, 0.1, 0.6, 0.7, 0.1})), 0))
	assert.Equal(t, Class(3), ToClass(T.New(T.WithBacking([]float64{0.1, 0.1, 0.6, 0.7, 0.1})), 0.65))
	assert.Equal(t, Class(0), ToClass(T.New(T.WithBacking([]float64{1, 1, 1, 1, 1})), 0))
	// nI
	assert.Equal(t, Class(2), ToClass(T.New(T.WithBacking([]int{0, 0, 1, 0, 0})), 0))
	assert.Equal(t, Class(2), ToClass(T.New(T.WithBacking([]int{0, 0, 1, 0, 0})), 999))
	assert.Equal(t, Class(2), ToClass(T.New(T.WithBacking([]int{0, 0, 2, 0, 0})), 0))
	assert.Equal(t, Class(1), ToClass(T.New(T.WithBacking([]int{0, 1, 2, 0, 0})), 0))
	assert.Equal(t, Class(0), ToClass(T.New(T.WithBacking([]int{1, 1, 1, 1, 1})), 0))
	assert.Equal(t, Class(4), ToClass(T.New(T.WithBacking([]int{-1, -2, -3, 0, 1})), 0))
	// iut
	assert.Equal(t, Class(2), ToClass(T.New(T.WithBacking([]uint{0, 0, 1, 0, 0})), 0))
	assert.Equal(t, Class(2), ToClass(T.New(T.WithBacking([]uint{0, 0, 1, 0, 0})), 999))
	assert.Equal(t, Class(2), ToClass(T.New(T.WithBacking([]uint{0, 0, 2, 0, 0})), 0))
	assert.Equal(t, Class(1), ToClass(T.New(T.WithBacking([]uint{0, 1, 2, 0, 0})), 0))
	assert.Equal(t, Class(0), ToClass(T.New(T.WithBacking([]uint{1, 1, 1, 1, 1})), 0))
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
	assert.Equal(t, []Class{2, 0}, ToClasses(matF32, 0))

	matF32 = T.New(T.WithBacking([]float32{0, 0, 1, 0, 1, 0}), shp)
	assert.Equal(t, []Class{2, 1}, ToClasses(matF32, 0))

	matF32 = T.New(T.WithBacking([]float32{0.1, 0.6, 0.7, 0.1, 0.7, 0.6}), shp)
	assert.Equal(t, []Class{1, 1}, ToClasses(matF32, 0))

	matF32 = T.New(T.WithBacking([]float32{0.1, 0.6, 0.7, 0.1, 0.7, 0.6}), shp)
	assert.Equal(t, []Class{2, 1}, ToClasses(matF32, 0.65))

	matF32 = T.New(T.WithBacking([]float32{0.1, -1, 0.7, -1.0, 0.2, 0.6}), shp)
	assert.Equal(t, []Class{2, 2}, ToClasses(matF32, 0))

	matF32 = T.New(T.WithBacking([]float32{1, 1, 1, 1, 1, 1}), shp)
	assert.Equal(t, []Class{0, 0}, ToClasses(matF32, 0))

	// Float64
	// chewxy testcase
	matF64 = T.New(T.WithBacking([]float64{0.1, 0.1, 0.6, 0.7, 0.1, 0.6, 0.1, 0.1, 0.7, 0.1}), T.WithShape(2, 5))
	assert.Equal(t, []Class{2, 0}, ToClasses(matF64, 0))

	matF64 = T.New(T.WithBacking([]float64{0, 0, 1, 0, 1, 0}), shp)
	assert.Equal(t, []Class{2, 1}, ToClasses(matF64, 0))

	matF64 = T.New(T.WithBacking([]float64{0.1, 0.6, 0.7, 0.1, 0.7, 0.6}), shp)
	assert.Equal(t, []Class{1, 1}, ToClasses(matF64, 0))

	matF64 = T.New(T.WithBacking([]float64{0.1, 0.6, 0.7, 0.1, 0.7, 0.6}), shp)
	assert.Equal(t, []Class{2, 1}, ToClasses(matF64, 0.65))

	matF64 = T.New(T.WithBacking([]float64{0.1, -1, 0.7, -1.0, 0.2, 0.6}), shp)
	assert.Equal(t, []Class{2, 2}, ToClasses(matF64, 0))

	matF64 = T.New(T.WithBacking([]float64{1, 1, 1, 1, 1, 1}), shp)
	assert.Equal(t, []Class{0, 0}, ToClasses(matF64, 0))

	// Int
	matInt = T.New(T.WithBacking([]int{0, 0, 1, 0, 1, 0}), shp)
	assert.Equal(t, []Class{2, 1}, ToClasses(matInt, 0))

	matInt = T.New(T.WithBacking([]int{0, 0, 1, 0, 1, 0}), shp)
	assert.Equal(t, []Class{2, 1}, ToClasses(matInt, 999))

	matInt = T.New(T.WithBacking([]int{0, 0, 2, 1, 2, 0}), shp)
	assert.Equal(t, []Class{2, 0}, ToClasses(matInt, 0))

	matInt = T.New(T.WithBacking([]int{1, 1, 1, 1, 1, 1}), shp)
	assert.Equal(t, []Class{0, 0}, ToClasses(matInt, 0))

	matInt = T.New(T.WithBacking([]int{-1, 1, -2, -3, 0, 1}), shp)
	assert.Equal(t, []Class{1, 2}, ToClasses(matInt, 0))

	// Uint
	matUint = T.New(T.WithBacking([]uint{0, 0, 1, 0, 1, 0}), shp)
	assert.Equal(t, []Class{2, 1}, ToClasses(matUint, 0))

	matUint = T.New(T.WithBacking([]uint{0, 0, 1, 0, 1, 0}), shp)
	assert.Equal(t, []Class{2, 1}, ToClasses(matUint, 999))

	matUint = T.New(T.WithBacking([]uint{0, 0, 2, 1, 2, 0}), shp)
	assert.Equal(t, []Class{2, 0}, ToClasses(matUint, 0))

	matUint = T.New(T.WithBacking([]uint{1, 1, 1, 1, 1, 1}), shp)
	assert.Equal(t, []Class{0, 0}, ToClasses(matUint, 0))
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
	assert.Equal(suite.T(), suite.expected.Data(), oh.Data())
	assert.Equal(suite.T(), suite.expected.Shape(), oh.Shape())
	if suite.unsafe {
		// Check if the operation is infact unsafe
		assert.Equal(suite.T(), suite.expected.Data(), suite.reuse.Data())
		assert.Equal(suite.T(), suite.expected.Shape(), suite.reuse.Shape())
		assert.Equal(suite.T(), &suite.reuse, &oh)
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

func NewToOneHotMatrixSuite(unsafe bool, a []Class, numClasses uint, backingActual, backingExpected interface{}, shp T.Shape) *ToOneHotMatrixSuite {
	return &ToOneHotMatrixSuite{
		unsafe:     unsafe,
		a:          a,
		numClasses: numClasses,
		reuse:      T.New(T.WithBacking(backingActual), T.WithShape(shp...)),
		expected:   T.New(T.WithBacking(backingExpected), T.WithShape(shp...)),
	}
}

// ToOneHotMatrixSuite test both the safe and unsafe version and the
// ToOneHotMatrix by specifying the `unsafe` boolean flag
type ToOneHotMatrixSuite struct {
	suite.Suite
	unsafe          bool
	a               []Class
	numClasses      uint
	reuse, expected *T.Dense
}

func (suite *ToOneHotMatrixSuite) Test() {
	// Safe or unsafe function
	var oh *T.Dense
	if suite.unsafe {
		oh = UnsafeToOneHotMatrix(suite.a, suite.numClasses, suite.reuse)
	} else {
		oh = ToOneHotMatrix(suite.a, suite.numClasses, suite.reuse.Dtype())
	}
	// Check data and shape between expected and resulting of UnsafeToOneHotMatrix
	assert.Equal(suite.T(), suite.expected.Data(), oh.Data())
	assert.Equal(suite.T(), suite.expected.Shape(), oh.Shape())
	if suite.unsafe {
		// Check if the operation is infact unsafe
		assert.Equal(suite.T(), suite.reuse.Data(), suite.expected.Data())
		assert.Equal(suite.T(), suite.reuse.Shape(), suite.expected.Shape())
		assert.Equal(suite.T(), &oh, &suite.reuse)
	}
}

func TestToOneHotMatrix(t *testing.T) {
	// Not Implemented
}

func TestUnsafeToOneHotMatrix(t *testing.T) {
	// suite.Run(t, NewToOneHotMatrixSuite(true, []Class{1}, 5, []float32{0, 0, 0, 0, 0}, []float32{0, 1, 0, 0, 0}, []int{1, 5}))
	// suite.Run(t, NewToOneHotMatrixSuite(true, []Class{1, 1}, 5, []float32{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, []float32{0, 1, 0, 0, 0, 0, 1, 0, 0, 0}, []int{2, 5}))
}
