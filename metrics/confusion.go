package metrics

import (
	"fmt"

	"gonum.org/v1/plot"
	"gorgonia.org/qol"
	qp "gorgonia.org/qol/plot"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/native"
)

// ConfusionMatrix represents a confusion matrix.
//
// It is ordered in the following shape (predicted, actual)
type ConfusionMatrix struct {
	*tensor.Dense
	classes qol.Classes
	Labels  []string    // optional Labels
	Iter    [][]float64 // a native iterator of the confusion matrix. This allows for quick access.
}

// Confusion creates a Confusion Matrix.
func Confusion(pred, correct qol.Classes) *ConfusionMatrix {
	classes := correct.Clone()
	classes = append(classes, pred...)
	classes = classes.Distinct()

	data := tensor.New(tensor.Of(tensor.Float64), tensor.WithShape(len(classes), len(classes)))
	iter, err := native.MatrixF64(data)
	if err != nil {
		panic(err)
	}
	for i := range correct {
		c := correct[i]
		p := pred[i]

		iter[p][c]++
	}
	return &ConfusionMatrix{
		classes: classes,
		Dense:   data,
		Iter:    iter,
	}
}

// Heatmap is a convenience method to create a heatmap.
// It creates a *gonum.org/v1/plot.Plot, which contains a heatmap structure.
// The Palette, X, Y are all accessible for customization.
//
// Not safe to be run concurrently.
func (m *ConfusionMatrix) Heatmap() (*plot.Plot, error) {
	labels := m.Labels
	if len(labels) == 0 {
		labels = make([]string, len(m.classes))
		for i := range labels {
			labels[i] = fmt.Sprintf("Class %d", i)
		}
	}
	return qp.Heatmap(m.Dense, labels)
}
