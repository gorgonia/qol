package plot

import (
	"image/color"
	"math"

	"github.com/pkg/errors"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/palette"
	"gonum.org/v1/plot/plotter"
	"gorgonia.org/qol/internal/tensorutils"
	"gorgonia.org/tensor"
)

// internal data structure to hold a heatmap
type heatmap struct {
	data mat.Matrix
}

func (m heatmap) Dims() (c, r int)   { r, c = m.data.Dims(); return c, r }
func (m heatmap) Z(c, r int) float64 { return m.data.At(r, c) }
func (m heatmap) X(c int) float64    { return float64(c) }
func (m heatmap) Y(r int) float64    { return float64(r) }

// internal data structure for heatmap ticks
type ticks []string

func (t ticks) Ticks(min, max float64) []plot.Tick {
	var retVal []plot.Tick
	for i := math.Trunc(min); i <= max; i++ {
		retVal = append(retVal, plot.Tick{Value: i, Label: t[int(i)]})
	}
	return retVal
}

func Heatmap(x tensor.Tensor, labels []string) (p *plot.Plot, err error) {
	pal := palette.Heat(48, 1)
	dense := tensorutils.GetDense(x)
	mat, err := tensor.ToMat64(dense, tensor.UseUnsafe())
	if err != nil {
		return nil, errors.Wrapf(err, "Converting a x (of type %T) to a mat.Dense failed.", x)
	}

	m := heatmap{mat}
	hm := plotter.NewHeatMap(m, pal)
	p = plot.New()
	hm.NaN = color.RGBA{0, 0, 0, 0} // black for NaN

	p.Add(hm)
	p.X.Tick.Label.Rotation = 1.5
	p.X.Tick.Marker = ticks(labels)
	p.Y.Tick.Marker = ticks(labels)

	return
}
