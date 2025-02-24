package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

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

func (nn *NeuralNet) train() error {
	wHidden := nn.glorotInit(nn.config.inputNeurons, nn.config.hiddenNeurons)
	bHidden := mat.NewDense(1, nn.config.hiddenNeurons, nil)
	wOut := nn.glorotInit(nn.config.hiddenNeurons, nn.config.outputNeurons)
	bOut := mat.NewDense(1, nn.config.outputNeurons, nil)

	nn.weightsHidden = wHidden
	nn.biasesHidden = bHidden
	nn.weightsOut = wOut
	nn.biasesOut = bOut
	return nil
}

func main() {
	config := NeuralNetConfig{
		inputNeurons:  4,
		outputNeurons: 3,
		hiddenNeurons: 3,
		numEpochs:     5000,
		learningRate:  0.3,
	}

	nn := newNetwork(config)

	nn.train()
	fmt.Println("Hidden Weights:")
	fmt.Println(mat.Formatted(nn.weightsHidden))
	fmt.Println("Output Weights:")
	fmt.Println(mat.Formatted(nn.weightsOut))
	fmt.Println("Hidden Biases:")
	fmt.Println(mat.Formatted(nn.biasesHidden))
	fmt.Println("Output Biases:")
	fmt.Println(mat.Formatted(nn.biasesOut))
}
