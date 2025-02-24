package main

import (
	"encoding/csv"
	"errors"
	"log"
	"math"
	"math/rand"
	"os"
	"time"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type NeuralNetConfig struct {
	inputNeurons  int
	outputNeurons int
	hiddenNeurons int
	numEpochs     int
	learningRate  float64
}

type NeuralNet struct {
	config        NeuralNetConfig
	weightsHidden *mat.Dense
	biasesHidden  *mat.Dense
	weightsOut    *mat.Dense
	biasesOut     *mat.Dense
}

func newNetwork(config NeuralNetConfig) *NeuralNet {
	return &NeuralNet{config: config}
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidPrime(x float64) float64 {
	return sigmoid(x) * (1.0 - sigmoid(x))
}

func (nn *NeuralNet) glorotInit(fanIn int, fanOut int) *mat.Dense {
	src := rand.NewSource(time.Now().UnixNano())
	rng := rand.New(src)

	limit := math.Sqrt(6.0 / float64(fanIn+fanOut))

	data := make([]float64, fanIn*fanOut)
	for i := range data {
		data[i] = (rng.Float64()*2 - 1) * limit
	}

	return mat.NewDense(fanIn, fanOut, data)
}

func (nn *NeuralNet) train(x, y *mat.Dense) error {
	wHidden := nn.glorotInit(nn.config.inputNeurons, nn.config.hiddenNeurons)
	bHidden := mat.NewDense(1, nn.config.hiddenNeurons, nil)
	wOut := nn.glorotInit(nn.config.hiddenNeurons, nn.config.outputNeurons)
	bOut := mat.NewDense(1, nn.config.outputNeurons, nil)

	output := new(mat.Dense)

	nn.forwardpropagate(x, y, wHidden, bHidden, wOut, bOut, output)

	nn.weightsHidden = wHidden
	nn.biasesHidden = bHidden
	nn.weightsOut = wOut
	nn.biasesOut = bOut
	return nil
}

func (nn *NeuralNet) forwardpropagate(x, y, wHidden, bHidden, wOut, bOut, output *mat.Dense) error {
	for i := 0; i < nn.config.numEpochs; i++ {
		hiddenLayerActivations := activation(x, wHidden, bHidden)
		outputLayerResult := outputComputation(hiddenLayerActivations, wOut, bHidden)
		err := lossCalculation(y, output)

		hiddenGradient := gradient(hiddenLayerActivations)
		outputGradient := gradient(outputLayerResult)

		nn.backpropagate(x, err, outputGradient, hiddenGradient, hiddenLayerActivations, wOut, wHidden, bHidden)

	}
	return nil
}

func (nn *NeuralNet) backpropagate(x, err, outputGradient, hiddenGradient, hiddenLayerActivations, wOut, wHidden, bHidden *mat.Dense) error {
	deltaOutput := new(mat.Dense)
	deltaOutput.MulElem(err, outputGradient)
	errorAtHiddenLayer := new(mat.Dense)
	errorAtHiddenLayer.Mul(deltaOutput, wOut.T())

	deltaHidden := new(mat.Dense)
	deltaHidden.MulElem(errorAtHiddenLayer, hiddenGradient)

	nn.adjustParams(hiddenLayerActivations, deltaOutput, wOut)
	hiddenBiasAdj, _ := nn.adjustParams(x, deltaHidden, wHidden)

	hiddenBiasAdj.Scale(nn.config.learningRate, hiddenBiasAdj)
	hiddenBiasAdj.Add(bHidden, hiddenBiasAdj)

	return nil
}

func (nn *NeuralNet) predict(x *mat.Dense) (*mat.Dense, error) {
	if nn.weightsHidden == nil || nn.weightsOut == nil {
		return nil, errors.New("the supplied weights are empty")
	}

	if nn.biasesHidden == nil || nn.biasesOut == nil {
		return nil, errors.New("the supplied biases are empty")
	}
	output := new(mat.Dense)

	hiddenLayerInput := new(mat.Dense)
	hiddenLayerInput.Mul(x, nn.weightsHidden)
	addBHidden := func(_, col int, v float64) float64 { return v + nn.biasesHidden.At(0, col) }
	hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

	hiddenLayerActivations := new(mat.Dense)
	applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
	hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

	outputLayerInput := new(mat.Dense)
	outputLayerInput.Mul(hiddenLayerActivations, nn.weightsOut)
	addBOut := func(_, col int, v float64) float64 { return v + nn.biasesOut.At(0, col) }
	outputLayerInput.Apply(addBOut, outputLayerInput)
	output.Apply(applySigmoid, outputLayerInput)

	return output, nil
}

func (nn *NeuralNet) adjustParams(input, delta, weights *mat.Dense) (*mat.Dense, error) {
	weightAdj := new(mat.Dense)
	weightAdj.Mul(input.T(), delta)
	weightAdj.Scale(nn.config.learningRate, weightAdj)
	weights.Add(weights, weightAdj)

	biasAdj, err := sumAlongAxis(0, delta)
	if err != nil {
		return nil, err
	}
	return biasAdj, nil
}

func sumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {
	numRows, numCols := m.Dims()

	var output *mat.Dense

	switch axis {
	case 0:
		data := make([]float64, numCols)
		for i := 0; i < numCols; i++ {
			col := mat.Col(nil, i, m)
			data[i] = floats.Sum(col)
		}
		output = mat.NewDense(1, numCols, data)
	case 1:
		data := make([]float64, numRows)
		for i := 0; i < numRows; i++ {
			row := mat.Row(nil, i, m)
			data[i] = floats.Sum(row)
		}
		output = mat.NewDense(numRows, 1, data)
	default:
		return nil, errors.New("invalid axis, must be 0 or 1")
	}

	return output, nil
}

func gradient(result *mat.Dense) *mat.Dense {
	slopeLayer := new(mat.Dense)
	slopeLayer.Apply(applySigmoidPrime, result)
	return slopeLayer
}

func lossCalculation(y, output *mat.Dense) *mat.Dense {
	networkError := new(mat.Dense)
	networkError.Sub(y, output)

	return networkError
}

func outputComputation(hiddenLayerActivations, weights, bias *mat.Dense) *mat.Dense {
	layer := new(mat.Dense)
	layer.Mul(hiddenLayerActivations, weights)
	addBias := func(_, col int, v float64) float64 {
		return v + bias.At(0, col)
	}
	layer.Apply(addBias, layer)
	layer.Apply(applySigmoid, layer)
	return layer
}

func activation(x, weights, bias *mat.Dense) *mat.Dense {
	layer := new(mat.Dense)
	layer.Mul(x, weights)
	addBias := func(_, col int, v float64) float64 {
		return v + bias.At(0, col)
	}
	layer.Apply(addBias, layer)

	layerActivations := new(mat.Dense)
	layerActivations.Apply(applySigmoid, layer)

	return layerActivations
}

func applySigmoid(_, _ int, v float64) float64 {
	return sigmoid(v)
}

func applySigmoidPrime(_, _ int, v float64) float64 {
	return sigmoidPrime(v)
}

func main() {
	f, err := os.Open("data/train.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	reader := csv.NewReader(f)
	reader.FieldsPerRecord = 11

	rawCSVData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}
}
