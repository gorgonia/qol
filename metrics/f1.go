package metrics

import (
	"math"

	"github.com/pkg/errors"
	"gorgonia.org/qol"
)

// F1 computes the balanced multiclass F1 score.
//
// It is computed with the following formula:
// 	F1 = 2 * (precision * recall) / (precision + recall)
func F1(pred, correct qol.Classes) (float64, error) {
	if len(pred) != len(correct) {
		return math.Inf(-1), errors.Errorf("Expected both predicted and corrects to be the same length. Predicted %d. Correct %d", len(pred), len(correct))
	}

	// find out how many classes there are
	classes := correct.Clone()
	classes = append(classes, pred...)
	classes = classes.Distinct()
	if len(classes) == 2 {
		// binary case
	}

	//var precision, recall float64
	panic("Unreachable")

}
