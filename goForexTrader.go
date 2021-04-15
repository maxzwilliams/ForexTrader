package main

// imports
import (
    "encoding/csv"
    "fmt"
    "log"
    "os"
    "unicode/utf8"
    "strconv"
    "strings"
    "time"
    "sort"
    "math"
    "gonum.org/v1/plot"
    "gonum.org/v1/plot/plotter"
    "math/rand"
    "sync"
    "flag"
    "runtime"
)

// custom types
type candle struct {
  t string
  openBid float64
  openAsk float64
  closeBid float64
  closeAsk float64
  highBid float64
  lowBid float64
  highAsk float64
  lowAsk float64
}

// data processing section (not needed to be written effeciently)
func readCsvFile(filePath string) [][]string {
    f, err := os.Open(filePath)
    if err != nil {
        log.Fatal("Unable to read input file " + filePath, err)
    }
    defer f.Close()

    csvReader := csv.NewReader(f)
    records, err := csvReader.ReadAll()
    if err != nil {
        log.Fatal("Unable to parse file as CSV for " + filePath, err)
    }

    return records
}
// might want to add an inverse option to this
func readCsvFiles(filePaths []string) [][]string {
  rtn := make([][]string, 0)
  for _, v := range filePaths{
    fileData := readCsvFile(v)
    for _, v2 := range fileData{
      rtn = append(rtn, v2)
    }
  }
  return rtn
}

func cleanCsvDataDict(data [][]string, period string) map[string]([][]float64){
  returnMap := make(map[string]([][]float64))
  var cutoff int
  if (period == "m"){
    cutoff = 5
  } else {
    fmt.Println("Wrong period given")
    time.Sleep(time.Second *1000)
  }
  var stringvalues []string
  var t string
  newEntry := make([]float64,0)
  baseEntry := make([][]float64,0)
  for _, value := range data{
    stringvalues = strings.Fields(value[0])
    t = stringvalues[0] + stringvalues[1][0:utf8.RuneCountInString(stringvalues[1])-cutoff]
    bid, _ := strconv.ParseFloat(stringvalues[2], 5)
    ask, _ := strconv.ParseFloat(stringvalues[3], 5)
    _, ok := returnMap[t]
    newEntry = []float64{bid, ask}
    if (ok == false){
      returnMap[t] = baseEntry
    }
    returnMap[t] = append(returnMap[t], newEntry)
  }
  return returnMap
}

func generateCandleData(cleanDict map[string]([][]float64)) []candle{
  rtn := make([]candle, 0)
  keys := make([]string,0, len(cleanDict))
  for kv := range cleanDict{
    keys = append(keys, kv)
  }
  sort.Strings(keys)
  spreadCounter := float64(0)
  // now keys are sorted
  var keyData [][]float64
  for _,kv := range keys{
    keyData = cleanDict[kv]
    openBid := keyData[0][0]
    openAsk := keyData[0][1]
    closeBid := keyData[len(keyData)-1][0]
    //closeAsk := closeBid*(1+0.01*0.01*0.50)
    closeAsk := keyData[len(keyData)-1][1]
    spreadCounter = spreadCounter + (closeAsk/closeBid - 1)/(0.01*0.01)

    highBid := float64(-1)
    lowBid := float64(9999)
    highAsk := float64(-1)
    lowAsk := float64(9999)

    for _,element := range keyData{
      cbid := element[0]
      cask := element[1]

      if (cbid > highBid){
        highBid = cbid
      }
      if (cbid < lowBid){
        lowBid = cbid
      }
      if (cask > highAsk){
        highAsk = cask
      }
      if (cask < lowAsk){
        lowAsk = cask
      }
    }
    cCandle := candle{t: kv, openBid: openBid, openAsk: openAsk, closeBid: closeBid, closeAsk: closeAsk, highBid: highBid, lowBid: lowBid, highAsk: highAsk, lowAsk: lowAsk}
    rtn = append(rtn, cCandle)
  }
  fmt.Println(spreadCounter/float64(len(keys)))
  time.Sleep(time.Second*10)
  //return rtn
  return setAVGSpread(rtn, 0.16, spreadCounter/float64(len(keys)))
}

func setAVGSpread(candleSeries []candle, desiredSpread float64, actualSpread float64) []candle {
  scale := desiredSpread/actualSpread
  fmt.Println("here is the scale", scale)
  rtn := make([]candle, 0)
  var newCandle candle
  for _, candle := range candleSeries{
    spread := ((candle.closeAsk/candle.closeBid - 1)/(0.01*0.01)) * scale
    newCandle.openAsk = candle.openBid*(1+0.01*0.01*spread)
    newCandle.closeAsk = candle.closeBid*(1+0.01*0.01*spread)

    newCandle.highAsk = candle.highBid*(1+0.01*0.01*spread)
    newCandle.lowAsk = candle.lowBid*(1+0.01*0.01*spread)

    newCandle.openBid = candle.openBid
    newCandle.closeBid = candle.closeBid
    newCandle.highBid = candle.highBid
    newCandle.lowBid = candle.lowBid
    rtn = append(rtn, newCandle)
  }
  return rtn
}

// could include the inverse stuff
func gatherCandleData(filenames []string, period string) []candle{
  rawCsvData := readCsvFiles(filenames)
  cleanDict := cleanCsvDataDict(rawCsvData, "m")
  candleData := generateCandleData(cleanDict)
  return candleData
}

func getTrainingAndTestingData(filenames []string, period string, trainingPercentage float64, headSize int, tailSize int, iteration int) ([]candle, [][][]candle){
  data := gatherCandleData(filenames, period)
  trainingData := data[0:int(math.Floor(float64(len(data)-1)*trainingPercentage))]
  testingData := data[int(math.Floor(float64(len(data)-1)*trainingPercentage)):len(data)-1]
  THTrainingData := convertToTHData(trainingData, headSize, tailSize, iteration)
  //fmt.Println(headSize)
  //fmt.Println(tailSize)
  return testingData, THTrainingData
}

func multiplyCandle(scalar float64, c candle) candle {
  kv := c.t
  openBid := c.openBid
  openAsk := c.openAsk
  closeBid := c.closeBid
  closeAsk := c.closeAsk
  highBid := c.highBid
  lowBid := c.lowBid
  highAsk := c.highAsk
  lowAsk := c.lowAsk
  newCandle := candle{t: kv, openBid: openBid*scalar, openAsk: openAsk*scalar, closeBid: closeBid*scalar, closeAsk: closeAsk*scalar, highBid: highBid*scalar, lowBid: lowBid*scalar, highAsk: highAsk*scalar, lowAsk: lowAsk*scalar}
  return newCandle
}

func multiplyCandleSeries(scalar float64, cs []candle) []candle{
  rtn := make([]candle, 0)
  for _, value := range cs{
    rtn = append(rtn, multiplyCandle(scalar, value))
  }
  return rtn
}

func addCandles(c1 candle, c2 candle) candle{
  var rtn candle
  rtn.t=c1.t
  rtn.openBid = c1.openBid + c2.openBid
  rtn.openAsk = c1.openAsk + c2.openAsk
  rtn.closeBid = c1.closeBid + c2.closeBid
  rtn.closeAsk = c1.closeAsk + c2.closeAsk
  rtn.highBid = c1.highBid + c2.highBid
  rtn.lowBid = c1.lowBid + c2.lowBid
  rtn.highAsk = c1.highAsk + c2.highAsk
  rtn.lowAsk = c1.lowAsk + c2.lowAsk
  return rtn
  }

func addCandleSeries(c1s []candle, c2s []candle) []candle{
  if (len(c1s) != len(c2s)){
    log.Fatal("trying to add candle series of different lengths")
  }
  rtn := make([]candle,0)
  for index, _ := range c1s{
    rtn = append(rtn, addCandles(c1s[index], c2s[index]))
  }
  return rtn
}

func normalizeCandles(candles []candle, normFactor float64) ([]candle, float64){
  if (normFactor == float64(-1)){
    normFactor = candles[len(candles)-1].closeBid
  }
  newCandles := make([]candle,0)
  for _, c := range candles{
    newCandles = append(newCandles, multiplyCandle(normFactor, c) )
  }
  return newCandles, normFactor
}

func convertToTHData(candleData []candle, headSize int, tailSize int, iteration int) [][][]candle{
  dataLength := len(candleData)
  segments := (dataLength-headSize-tailSize)/iteration + 1
  rtn := make([][][]candle, 0)
  for index := 0; index < segments; index ++{
    segment := candleData[index*iteration: index*iteration + headSize + tailSize]
    segmentHead, normFactor := normalizeCandles(segment[0:headSize], -1)
    segmentTail, _ := normalizeCandles(segment[headSize: headSize + tailSize], normFactor)
    entry := [][]candle{segmentHead, segmentTail}
    rtn = append(rtn, entry)
  }
  return rtn

}

type model struct{
  name string
  inputSize int
  outputSize int
  leverage float64
  network NN
}

func evaluateModel(m model, input []candle) []float64 {
  //rtn := make([]float64,0)
  //for index:=0;index<60;index++{
    //rtn = append(rtn, float64(1))
  //}
  rtn := []float64{0.015410294697045853, 0.015781519622597544, 0.014014280332988089, 0.014567047725310926, 0.01637919676948722, 0.01637472097452204, 0.015444378291903618, 0.01628118547734177, 0.016024267848130342, 0.016855467706696516, 0.01710070979498782, 0.016124898186321155, 0.01495888589834501, 0.015633496395549864, 0.015712164600605184, 0.015081207450702874, 0.01463157413750256, 0.014101200135525582, 0.015345690867997233, 0.01486213450841722, 0.014780642138737615, 0.015077545998680813, 0.014098587425724052, 0.013481569386235043, 0.012998889669467755, 0.014499483588596079, 0.014079965159794714, 0.013731886975967687, 0.014240225965566513, 0.01422274855911239, 0.014662850159752851, 0.015094963283479906, 0.01470235671850946, 0.013551307393422108, 0.015020048571178366, 0.013848131690286616, 0.013757714776350943, 0.013176615891301285, 0.014250558995109283, 0.015148661001562893, 0.013359519670457072, 0.014404377729832336, 0.01347580412775508, 0.014149721574794785, 0.013974972747057087, 0.013554901389374482, 0.01319196432410661, 0.01286782720927705, 0.01363945395116143, 0.014674862371467277, 0.015361649160855508, 0.01628206574740216, 0.01865892302760299, 0.019690699872761636, 0.02129193991740774, 0.024949301058573868, 0.02791202516509252, 0.03303725595468839, 0.04131643473547574, 0.04909722549404165}
  return rtn

}

type position struct{
  active bool
  value float64
  posType string
  entry float64
  length int
}

type modelState struct{
  currentPosition []position
}

func printCandle(c candle){
  fmt.Println("time",c.t,"openBid", c.openBid, "openAsk", c.openAsk, "closeBid", c.closeBid, "closeAsk", c.closeAsk, "highBid", c.highBid, "lowBid", c.lowBid, "highAsk", c.highAsk, "lowAsk", c.lowAsk )
}

// Prediction Section. Takes the pointers of the two series and weigths and returns the error
func candleSeriesError(cs1 *[]candle, cs2 *[]candle, weights *[]float64, cutoff float64) float64{
  //var error float64
  error := float64(0)
  s1 := *cs1
  s2 := *cs2
  w := *weights
  if (len(s1) != len(s2)){
    fmt.Println("seires not the same size")
    log.Fatal("series not the same size")
    time.Sleep(1000* time.Second)
  }
  if (len(s1) != len(w)){

    log.Fatal("Weights not the correct size " + strconv.Itoa(len(s1)) +" "+ strconv.Itoa(len(w)))

  }
  var c1 candle
  var c2 candle
  for index := range s1{
    c1 = s1[index]
    c2 = s2[index]
    error = error + ((c1.openBid-c2.openBid)*(c1.openBid-c2.openBid) + (c1.openAsk-c2.openAsk)*(c1.openAsk-c2.openAsk) + (c1.closeBid-c2.closeBid)*(c1.closeBid-c2.closeBid) + (c1.closeAsk-c2.closeAsk)*(c1.closeAsk-c2.closeAsk) + (c1.highBid-c2.highBid)*(c1.highBid-c2.highBid) + (c1.lowBid-c2.lowBid)*(c1.lowBid-c2.lowBid) + (c1.highAsk-c2.highAsk)*(c1.highAsk-c2.highAsk) + (c1.lowAsk-c2.lowAsk)*(c1.lowAsk-c2.lowAsk) )*w[index]
    if (math.IsNaN(error)){
      fmt.Println(c1)
      fmt.Println(c2)
      fmt.Println(w)
      log.Fatal("problem in candle error")
    }
    if (error > cutoff){
      return float64(99999)
    }
  }
  return error
}

func findMax(array []float64) (float64, int){
  maxValue := float64(-1)
  var maxIndex int
  for i, v := range array{
    if (v > maxValue){
      maxValue = v
      maxIndex = i
    }
  }
  return maxValue, maxIndex
}

func findMin(array []float64) (float64, int){
  minValue := float64(1.797693134862315708145274237317043567981e+308)
  var minIndex int
  for i, v := range array{
    if (v < minValue){
      minValue = v
      minIndex = i
    }
  }
  return minValue, minIndex
}

func remove(s []float64, i int) []float64 {
    s[len(s)-1], s[i] = s[i], s[len(s)-1]
    return s[:len(s)-1]
}


// returns predicted candles and an array for the uncertainty
// because go is really fast, lets assume that we can use the whole THData
func predict(currentSeries []candle, THData * [][][]candle, predictionPopulation int, weightMatrix []float64) []candle{
  pastData := *THData
  headLength := len(pastData[0][0])
  tailLength := len(pastData[0][1])
  //normFactor := currentSeries[len(currentSeries) - 1].closeBid
  nCurrentSeries, normFactor := normalizeCandles(currentSeries,-1)
  if (headLength != len(currentSeries)){
    log.Fatal("predict got incorrect data dimensions")
  }
  errorTailMap := make(map[string]([]candle)) //have to store the errors as strings to avoid weird shit
  var ferror float64
  var serror string
  var totalScore float64
  cutoff := float64(999999) //think about this
  lowestErrors := make([]float64, 0)
  for i :=0; i < predictionPopulation;i++{
    lowestErrors = append(lowestErrors, cutoff)
  }
  for _, dataPoint := range pastData{
    head := dataPoint[0]
    tail := dataPoint[1]
    cutoff, _ = findMax(lowestErrors)
    ferror = candleSeriesError(&head, &nCurrentSeries, &weightMatrix, cutoff)
    serror = strconv.FormatFloat(ferror, 'f', -1, 64)
    errorTailMap[serror] = tail
    totalScore = totalScore + 1.0/ferror

    if (ferror < cutoff){
      lowestErrors = append(lowestErrors, ferror)
      _, rIndex := findMax(lowestErrors)
      lowestErrors = remove(lowestErrors, rIndex)
    }
  }
  predictionTally := make([]candle, tailLength)
  // here is where we need the candle sum function
  //skeys := make([]string,0)
  newTotalScore := float64(0)
  for errorString := range errorTailMap{
    floatError, _ := strconv.ParseFloat(errorString, 64)
    predictionTally = addCandleSeries(multiplyCandleSeries(1.0/floatError, errorTailMap[errorString]), predictionTally)
    newTotalScore = newTotalScore + 1/floatError
  }
  predictionTally = multiplyCandleSeries(1.0/totalScore, predictionTally)
  predictionTally = multiplyCandleSeries(1/normFactor, predictionTally)
  return predictionTally
}

func simulateTrading(candleSeries []candle, THDataPointer * [][][]candle, predictionPopulation int, n NN) float64{
  THData := *THDataPointer
  headLength := len(THData[0][0])
  tailLength := len(THData[0][1])
  candleSeriesLength := len(candleSeries)
  var tickCandles []candle
  tally := float64(1)
  var pos position
  tallies := make([]float64, 0)
  final := false
  for index:=headLength;index<(candleSeriesLength-headLength-tailLength);index++{
    if ((index-headLength) % 100 == 0){
      fmt.Println("percent through is " + strconv.FormatFloat((float64(index-headLength))/float64((candleSeriesLength-headLength-tailLength)), 'f', -1, 64))
    }

    tickCandles = candleSeries[index:index+headLength]
    if (index ==(candleSeriesLength-headLength-tailLength)-1){
      final = true
    }
    pos, tally = simulateTick(tickCandles, THDataPointer, predictionPopulation, n, pos, tally, final)
    tallies = append(tallies, tally)
    if (len(tallies) > 2){
      if (tallies[len(tallies)-1] != tallies[len(tallies)-2]){
        fmt.Println(tallies[len(tallies)-1])
      }
    }
  }
  //n := rand.Int()
  //plotArray("accountWorth"+strconv.Itoa(n)+".png", tallies)
  //plotCandleData("candleSeries"+strconv.Itoa(n)+".png", candleSeries, "closeBid")
  return tally
}

// take the current series, a model and our current position and trade
// candleSeries must be exactly the write size
func simulateTick(candleSeries []candle, THDataPointer * [][][]candle, predictionPopulation int, n NN, pos position, tally float64, final bool) (position, float64) {

  leverage := float64(100)
  rtCost := float64(7*0.65)/float64(100000)
  buffer := tally*leverage*rtCost*0
  //buffer := float64(0)
  maxLength := int(60)
  normed, _ := normalizeCandles(candleSeries, -1)
  weightMatrix := evaluate(n, candleStrip(normed))
  //fmt.Println(weightMatrix)
  //weightMatrix := evaluateModel(m, candleSeries)
  predictedSeries := predict(candleSeries, THDataPointer, predictionPopulation, weightMatrix)
  // bid is always lower than ask
  // you buy at the ask price and sell at the bid price
  // shorts enter at the bid price and close at the ask price
  // so shorts want their ask price to be lower than there bid price
  currentAsk := candleSeries[len(candleSeries)-1].closeAsk
  currentBid := candleSeries[len(candleSeries)-1].closeBid
  maxCloseBid := float64(-1)
  minCloseBid := float64(9999)
  maxCloseAsk := float64(-1)
  minCloseAsk := float64(9999)
  if (currentAsk <= currentBid){
    fmt.Println(currentAsk, currentBid)
    fmt.Println("current ask is less than current bid, something went wrong")
    time.Sleep(time.Second*100)
  }

  //randomAction := rand.Int() % 5

  var closeBid float64
  var closeAsk float64
  for _, el := range predictedSeries{
    closeBid = el.closeBid
    closeAsk = el.closeAsk
    if (closeBid > maxCloseBid){
      maxCloseBid = closeBid
    }
    if (closeBid < minCloseBid){
      minCloseBid = closeBid
    }
    if (closeAsk > maxCloseAsk){
      maxCloseAsk = closeAsk
    }
    if (closeAsk < minCloseAsk){
      minCloseAsk = closeAsk
    }
  }
  // we now have the min and max close prices through our prediction
  if (pos.active == false && final == false){
    //if (randomAction == 1){
    //&& currentAsk < minCloseAsk
    if (currentAsk < maxCloseBid && currentAsk < minCloseAsk){
      // consider the buffer
      pPercentChange := (maxCloseBid - currentAsk)/currentAsk
      pValue := tally*(1+leverage*pPercentChange)
      if (pValue - buffer*2 > tally){
        // buy
        pos.active = true
        pos.entry = currentAsk
        pos.posType = "buy"
        pos.value = tally
        pos.length = 0
      }
    //} else if (randomAction == 5){
    // && currentBid > maxCloseBid
    } else if (currentBid < minCloseAsk && currentBid > maxCloseBid){
      pPercentChange := (minCloseAsk - currentBid)/currentBid
      pValue := tally*(1+leverage*pPercentChange)
      if (pValue - buffer*2 > tally){
        pos.active = true
        pos.entry = currentBid
        pos.posType = "short"
        pos.value = tally
        pos.length = 0
      }
    }
    } else {
      pos.length = pos.length + 1
      if (pos.posType == "buy"){
        //if (randomAction == 1){
        if (currentBid > maxCloseBid || final==true || pos.length > maxLength){
          // then we want to sell
          percent := (currentBid-pos.entry)/pos.entry
          tally = (tally - pos.value) + pos.value*(1+percent*leverage)
          // commision
          tally = tally - buffer
          pos.active = false
        }
      } else if (pos.posType == "short"){
        //if (randomAction == 5){
        if (currentAsk < minCloseAsk || final == true || pos.length > maxLength){
          percent := (pos.entry - currentAsk)/pos.entry
          tally = (tally - pos.value) + pos.value*(1+percent*leverage)
          tally = tally - buffer
          pos.active = false
        }
      }
    }
    return pos, tally

}

// simulates predictons over a data series when running our genetic algorithm
func simulatePredictionsNew(candleSeriesPointer * []candle, THDataPointer * [][][]candle, predictionPopulation int, network NN, startIndex int) (float64,float64){
  candleSeries := *candleSeriesPointer
  THData := *THDataPointer
  headLength := len(THData[0][0])
  tailLength := len(THData[0][1])
  //dataLength := len(THData)
  var prediction []candle
  var realResult []candle
  //setting := "closeBid"
  errorTally := make([]float64,0)

  nWeightMatrix := make([]float64,0)
  for index:=0; index < tailLength; index++{
    nWeightMatrix = append(nWeightMatrix, float64(1))
  }
  //for index:=0;index<len(candleSeries)-headLength;index++{
  for index := startIndex; index < startIndex+100; index ++{
    predictionData, _ := normalizeCandles(candleSeries[index:index+headLength], -1)
    weightMatrix := evaluate(network, candleStrip(predictionData))
    //fmt.Println("here is the weight matrix", weightMatrix)
    prediction = predict(candleSeries[index:index+headLength], &THData, predictionPopulation, weightMatrix)
    realResult = candleSeries[index+headLength:index+headLength+tailLength]
    error := candleSeriesError(&prediction, &realResult, &nWeightMatrix, 9999999999)
    errorTally = append(errorTally, error)
  }
  //tally := simulateTrading(candleSeries, THDataPointer, predictionPopulation, network)
  tally := float64(1)
  return newArrayAverage(errorTally), tally
}


// nbeed takes a list of two parents and a random noise percent and produces one
// offspring
func breed(parents []NN, randomNoise float64) NN{
  var nNN NN
  p1 := parents[0]
  p2 := parents[1]
  //fmt.Println("starting matricies", p1.weightMatricies[0])
  //fmt.Println("starting matricies", p2.weightMatricies[0])
  if (len(parents) > 2){
    log.Fatal("giving nbreed the wrong parents")
  }
  if (len(parents) == 1){
    fmt.Println("breed failed, tried to breed with itself")
    return parents[0]
  }

  newBiasVectors := make([]vector, 0)
  for bIndex:=0;bIndex<len(p1.biasVectors);bIndex++{
    //fmt.Println("initial bias vectors")
    //fmt.Println(p1.biasVectors[bIndex])
    //fmt.Println(p2.biasVectors[bIndex])
    newBiasVector := vvAdd(p1.biasVectors[bIndex], p2.biasVectors[bIndex])
    newBiasVector = svMultiply(0.5, newBiasVector)
    deviationVector := vvAdd(p1.biasVectors[bIndex], svMultiply(-1.0, p2.biasVectors[bIndex]))
    deviationVector = vvDot(deviationVector, randomVector(deviationVector.length))
    deviationVector = svMultiply(randomNoise, deviationVector)
    newBiasVector = vvAdd(newBiasVector, deviationVector)
    newBiasVectors = append(newBiasVectors, newBiasVector)
    //fmt.Println("output vector")
    //fmt.Println(newBiasVector)
  }
  newMatrixes := make([]matrix, 0)
  for mIndex:=0;mIndex<len(p1.weightMatricies);mIndex++{
    newMatrix := mmAdd(p1.weightMatricies[mIndex], p1.weightMatricies[mIndex])
    deviationMatrix := mmAdd(p1.weightMatricies[mIndex], smMultiply(-1.0, p2.weightMatricies[mIndex]))
    deviationMatrix = mmDot(deviationMatrix, randomMatrix(deviationMatrix.rows, deviationMatrix.cols))
    deviationMatrix = smMultiply(randomNoise, deviationMatrix)
    newMatrix = smMultiply(0.5, newMatrix)
    newMatrix = mmAdd(newMatrix, deviationMatrix)
    //fmt.Println("after being added", newMatrix)
    newMatrixes = append(newMatrixes, newMatrix)
  }
  //fmt.Println("here are the new matricies", newMatrixes[0])
  nNN.weightMatricies = newMatrixes
  nNN.biasVectors = newBiasVectors
  nNN.activationFunctions = p1.activationFunctions
  return nNN
}

func generateInitialPopulation(populationSize int, layerSizes []int, activationFunctions []string) []NN{
  rtn := make([]NN, 0)
  for index:=0;index<populationSize;index++{
    rtn = append(rtn, generateRandomNetwork(layerSizes, activationFunctions))
  }
  return rtn
}

func getFitnessUN(individual NN, candleSeriesPointer * []candle, THDataPointer * [][][]candle, predictionPopulation int, startIndex int) float64{
  error, _ := simulatePredictionsNew(candleSeriesPointer, THDataPointer, predictionPopulation, individual, startIndex)
  //fmt.Println(tally)
  //fmt.Println(error)
  return float64(1)/error
}

// lets make a fast version of this
// we need to return the scores in order of the parents
func scorePopulation(parents []NN, candleSeriesPointer * []candle, THDataPointer * [][][]candle, predictionPopulation int) []float64{
  minIndex := 0
  maxIndex := len(*candleSeriesPointer) - len((*THDataPointer)[0][0]) - len((*THDataPointer)[0][1]) - 100
  startIndex := rand.Intn(maxIndex - minIndex) + minIndex
  //startIndex := 0
  parentToScore := sync.Map{}
  var wg sync.WaitGroup
  pointerList := make([] *NN, 0)
  for index:=0;index<len(parents);index++{
    pointerList = append(pointerList, &(parents[index]))
  }

  populationSize := len(parents)
  wg.Add(populationSize)
  for i:=0;i<populationSize;i++{
    go func(i int){
      score := getFitnessUN(parents[i], candleSeriesPointer, THDataPointer, predictionPopulation, startIndex)
      //fmt.Println("here is the score", score)
      parentToScore.Store(pointerList[i], score)
      wg.Done()
    }(i)
  }
  wg.Wait()

  dict := make(map[*NN]float64)
  pointers := make([]*NN, 0)

  parentToScore.Range(func(key, value interface{}) bool{
    val, ok := value.(float64)
    //fmt.Println("here is what is getting out", val)
    if !ok{
      return false
    }
    dict[key.(*NN)] = val
    pointers =append(pointers, key.(*NN))
    //fmt.Println("here is what is getting added", key.(*NN))
    //fmt.Println("and whats going there", val)
    //fmt.Println("and here is what the dict contains", dict[key.(*NN)])
    return true
  })
  rtn := make([]float64, 0)
  for _, el := range pointerList{
    //fmt.Println("trying to search for", el)
    //fmt.Println("getting this out", dict[el])
    rtn = append(rtn, dict[el])
  }
  //fmt.Println(rtn)
  return rtn
}

func selectBest(population []NN, scores []float64) ([]NN, []float64, NN){
  _, maxIndex := findMax(scores)
  newPopulation := make([]NN, 0)
  newScores := make([]float64, 0)
  for index:=0;index<len(population);index++{
    if (maxIndex != index){
      newPopulation = append(newPopulation, population[index])
      newScores = append(newScores, scores[index])
    }
  }
  return newPopulation, newScores, population[maxIndex]
}

func getBestScoredNetworks(population []NN, scores []float64, num int) []NN{
  bestNetworks := make([]NN, 0)
  var bestNet NN
  for index:=0;index<num;index++{
    population, scores, bestNet = selectBest(population, scores)
    bestNetworks = append(bestNetworks, bestNet)
  }
  return bestNetworks
}

func getBestIndex(scores []float64, pop []NN) ([]float64, []NN, float64, NN){
  fmt.Println(len(scores), len(pop))
  _, bestIndex := findMax(scores) // woop
  rtnPop := make([]NN, 0)
  rtnScores := make([]float64, 0)
  //rtnScores := make([]float64, 0)
  for index:=0;index<len(scores);index++{
    //fmt.Println("index", index, len(scores))
    if (index != int(bestIndex)){
      rtnPop = append(rtnPop, pop[index])
      rtnScores = append(rtnScores, scores[index])
    }

  }
  return rtnScores, rtnPop, scores[bestIndex], pop[bestIndex]
}

func nWeightedSelect(scores []float64, currentGeneration []NN) []NN {
  winnersPoolLength := int(float64(len(currentGeneration))*0.50)
  rtnPop := make([]NN, 0)
  rtnScores := make([]float64, 0)
  var score float64
  var network NN
  counter := 0
  for (counter < winnersPoolLength){
    scores, currentGeneration, score, network =  getBestIndex(scores, currentGeneration)
    rtnPop = append(rtnPop, network)
    rtnScores = append(rtnScores, score)
    counter++
  }
  //scores = make([]float64, 0)

  return weightedSelect(rtnScores, rtnPop, 2)

}

func generateNextGeneration(currentGeneration []NN, candleSeriesPointer * []candle, THDataPointer * [][][]candle, predictionPopulation int, percentRetain float64, randomNoise float64) ([]NN,float64 ,float64 ,float64 , NN){
  scores := scorePopulation(currentGeneration, candleSeriesPointer, THDataPointer, predictionPopulation)
  //fmt.Println("scores list", scores)
  //scores2 := scorePopulatioOld(currentGeneration,candleSeriesPointer,THDataPointer,predictionPopulation)
  //fmt.Println(scores)
  //fmt.Println(scores2)
  eliteNumber := int(math.Floor(float64(len(currentGeneration))*0.10))
  currentPop := eliteNumber
  rtn := getBestScoredNetworks(currentGeneration, scores, eliteNumber)

  _, index := findMax(scores) // woop
  _, lowIndex := findMin(scores)
  //rtn = append(rtn, currentGeneration[index])
  for (currentPop < int(math.Floor(float64(len(currentGeneration))*percentRetain))){
    selectedPop := nWeightedSelect(scores, currentGeneration)
    rtn = append(rtn, breed(selectedPop, randomNoise))
    currentPop = currentPop + 1
  }
  //fmt.Println("here is the generations current score", arrayAverage(scores, 9999999))
  return rtn, scores[index], scores[lowIndex], newArrayAverage(scores), currentGeneration[index]
}

func printOutNN(n NN){
  fmt.Println("bias vectors")
  for _, el := range n.biasVectors{
    fmt.Println(el.values)
  }
  fmt.Println("activation functions")
  for _, el := range n.activationFunctions{
    fmt.Println(el)
  }
  fmt.Println("matricies")
  for _, el := range n.weightMatricies{
    fmt.Println(el.values)
  }
}

// main function from which the genetic algorithm is run
func GTrain(initialPopSize int, randomNoise float64, percentRetain float64, layerSizes []int, activationFunctions []string, candleSeriesPointer * []candle, THDataPointer * [][][]candle, predictionPopulation int){
  currentPop := generateInitialPopulation(initialPopSize, layerSizes, activationFunctions)
  var bestNetwork NN
  var score float64
  var worstScore float64

  counter := 1
  scores := make([]float64, 0)
  bestScores := make([]float64, 0)
  smoothedScores := make([]float64, 0)
  smoothedBestScores := make([]float64, 0)
  superSmoothedScores := make([]float64, 0)


  for (len(currentPop) > 1){
    bestScore := float64(-1)
    bestScore = bestScore - float64(1)
    start := time.Now()
    currentPop, bestScore, worstScore ,score, bestNetwork = generateNextGeneration(currentPop, candleSeriesPointer, THDataPointer, predictionPopulation, percentRetain, randomNoise)
    //worstScore, _ := findMin(scores)

    fmt.Println("here is the current population", len(currentPop))
    fmt.Println("here is the average score", score)
    fmt.Println("here is the best score", bestScore)
    fmt.Println("here is the worst score", worstScore)
    fmt.Println("percent deviation in probability of selection", (bestScore - worstScore)/worstScore )
    fmt.Println("epoch number", counter)
    fmt.Println("epoch took", time.Since(start))
    fmt.Println("here is the best network of the epoch")
    printOutNN(bestNetwork)
    scores = append(scores, score)
    fmt.Println("here are the scores", scores)
    bestScores = append(bestScores, score)
    Nscores := scores
    NNscores := scores
    NbestScores := bestScores
    if (len(scores) > 100){
      NNscores =  scores[len(scores)-1-100:len(scores)-1]
    }
    if (len(scores) > 50){
      Nscores = scores[len(scores)-1-50:len(scores)-1]
    }
    if (len(bestScores) > 10){
      NbestScores = bestScores[len(bestScores)-1-10:len(bestScores)-1]
    }

    smoothedScores = append(smoothedScores, newArrayAverage(Nscores))
    superSmoothedScores = append(superSmoothedScores, newArrayAverage(NNscores))
    smoothedBestScores = append(smoothedBestScores, newArrayAverage(NbestScores))
    counter = counter + 1
    plotArray("epochAVG.png", scores)
    plotArray("epochBEST.png", bestScores)
    plotArray("epochSUPERSMOOTH.png", superSmoothedScores)
    plotArray("epochAVGSMOOTHED.png", smoothedScores)
    plotArray("epochBESTSMOOTHED.png", smoothedBestScores)
  }
  //fmt.Println(getFitnessUN(currentPop[0], candleSeriesPointer, THDataPointer, predictionPopulation))
  printOutNN(currentPop[0])
  fmt.Println("finnished genetic training")
}


func multiplyArray(scalar float64, array []float64) []float64{
  for i := range array{
    array[i] = array[i]*scalar
  }
  return array
}

func addArray(array1 []float64, array2 []float64) []float64{
  rtn := make([]float64, 0)
  var newEntry float64
  for index, _ := range array1{
    newEntry = array1[index] + array2[index]
    rtn = append(rtn, newEntry)
  }
  return rtn
}

func normalizeArray(array []float64) []float64{
  sum := float64(0)
  sumArray:=make([]float64, 0)
  for _, v := range array{
    sum = sum + v
    sumArray = append(sumArray, sum)
    //fmt.Println(sum)
  }
  if (sum == 0){
    rtn:=make([]float64, 0)
    for index:=0;index<len(array);index++{
      rtn = append(rtn, float64(1)/float64(len(array)))
    }
    return rtn
  }
  rtn := multiplyArray(float64(1)/sum, array)
  if (math.IsNaN(rtn[0])){
    fmt.Println("here is the array")
    fmt.Println(array)
    fmt.Println(sumArray)
    fmt.Println(sum)
    fmt.Println(1/sum)
    log.Fatal("normalizeArray is wrong")
  }
  return rtn
}

func generateRandomNormalizedArray(length int) []float64{
  rtn := make([]float64, 0)
  for index:=0;index<length;index++{
    rtn = append(rtn, math.Exp(float64(10)*rand.Float64()))
  }
  return normalizeArray(rtn)
}

func arrayAverage(array []float64, period int) float64{
  sum := float64(0)
  for i, v := range array{
    if (len(array)-i < period){
      sum = sum + v
    }
  }
  if (len(array) < period){
    return sum/float64(len(array))
  }
  return sum/float64(period)
}

func newArrayAverage(array []float64) float64{
  sum := float64(0)
  for _, v := range array{
    sum = sum + v
  }
  return sum/float64(len(array))
}


// function for getting the refined candleDataList for input into neural networks
// currently this is just hardcoded for picking out the closeBids as that is all that we need right now
func candleStrip(candles []candle) []float64{
  rtn := make([]float64, 0)
  for _, v := range candles{
    rtn = append(rtn, v.closeBid)
  }
  return rtn
}




// convert our array of candles to a set of XY pointers. We will plot the property indicated by the setting string
func convertCandlesToPlotter(array []candle, setting string) plotter.XYs{
  if (setting != "closeBid"){
    log.Fatal("not coded to plot that type")
  }
  rtn := make(plotter.XYs, 0)
  var newXY plotter.XY
  for index:=0; index < len(array); index++{
    newXY.X = float64(index)
    newXY.Y = array[index].closeAsk
    rtn = append(rtn, newXY)
  }
  return rtn
}

func convertArrayToPlotter(array []float64) plotter.XYs{
  rtn := make(plotter.XYs,0)
  var newXY plotter.XY
  for index:=0;index<len(array);index++{
    newXY.X = float64(index)
    newXY.Y = array[index]
    rtn = append(rtn, newXY)
  }
  return rtn
}

func plotArray(path string, array []float64){
  xys := convertArrayToPlotter(array)
  f, _ := os.Create(path)
  p, _ := plot.New()
  s, _ := plotter.NewScatter(xys)
  p.Add(s)
  wt, _ := p.WriterTo(512, 512, "png")
  wt.WriteTo(f)
}
// takes a candle array and a setting and plots
func plotCandleData(path string, candleArray []candle, setting string){
  xys := convertCandlesToPlotter(candleArray, setting)
  f, _ := os.Create(path)
  p, _ := plot.New()
  s, _ := plotter.NewScatter(xys)
  p.Add(s)
  wt, _ := p.WriterTo(512, 512, "png")
  wt.WriteTo(f)
}





// something is really broken with the NN stuff



// neural network stuff
type NN struct{
  biasVectors []vector
  activationFunctions []string
  weightMatricies []matrix
}

type vector struct{
  values []float64
  length int
}

type matrix struct{
  values [][]float64
  rows int
  cols int
}

func generateRandomRow(rowLength int) []float64{
  rtn := make([]float64, 0)
  for index:=0;index<rowLength;index++{
    rtn = append(rtn, 0.01-0.02*rand.Float64())
  }
  return rtn
}

// make a random matrix of given size
func randomMatrix(rows int, cols int) matrix{
  var m matrix
  values := make([][]float64,0)
  for index:=0;index<rows;index++{
    values = append(values, generateRandomRow(cols))
  }

  m.rows = rows
  m.cols = cols
  m.values = values
  //fmt.Println("generating weight matrixes", values)
  return m
}

// have to write a random vector function aswell
func randomVector(length int) vector{
  var v vector
  v.length = length
  values := make([]float64, length)
  for index:=0;index<length;index++{
    values[index] = 0.01 - 0.02*rand.Float64()
  }
  v.values = values
  return v
}

//generate a random network
func generateRandomNetwork(layerDimensions []int, activationFunctions []string) NN{
  biasVectors := make([]vector, 0)
  weightMatricies := make([]matrix, 0)
  for i,_ := range layerDimensions{
    biasVectors = append(biasVectors, randomVector(layerDimensions[i]) )
    if (i != 0){
      weightMatricies = append(weightMatricies, randomMatrix(layerDimensions[i], layerDimensions[i-1]))
    }
  }
  var n NN
  n.biasVectors = biasVectors
  n.activationFunctions = activationFunctions
  n.weightMatricies = weightMatricies
  return n
}


// basic matrix-vector operations
// matrix-vector multiply appears to work fine
func mvMultiply(m matrix, v vector) vector{
  if (m.cols != v.length){
    log.Fatal("Matrix-Vector multiply got entries of the wrong dimensions. Matrix had cols " + strconv.Itoa(m.cols) + " and vector had length" + strconv.Itoa(v.length))
  }

  var rv vector
  values := make([]float64, m.rows)
  for index:=0; index < m.rows; index++{
    values[index]=0
    for innerIndex:=0;innerIndex<v.length;innerIndex++{
      values[index] = values[index] + m.values[index][innerIndex]*v.values[innerIndex]
    }
  }
  rv.length = m.rows
  rv.values = values
  return rv
}

// vector-vector add // checked
func vvAdd(v1 vector, v2 vector) vector {
  if (v1.length != v2.length){
    log.Fatal("Vector-Vector add has two vectors of the wrong size " + strconv.Itoa(v1.length) + " and " + strconv.Itoa(v2.length))
  }
    var rv vector
    values := make([]float64, v1.length)
    for index:=0;index<v1.length;index++{
      values[index] = v1.values[index] + v2.values[index]
    }
    rv.length = v1.length
    rv.values = values
    return rv
}

func svMultiply(scalar float64, v vector) vector{
  var nv vector
  nv.length = v.length
  values := multiplyArray(scalar, v.values)
  nv.values = values
  return nv
}

func mmDot(m1 matrix, m2 matrix) matrix{
  var m matrix
  if (m1.cols != m2.cols || m1.rows != m2.rows){
    log.Fatal("dot product got different sized things")
  }
  values := m1.values
  for outerIndex:=0;outerIndex<len(m1.values);outerIndex++{
    for innerIndex:=0;innerIndex<len(m1.values[outerIndex]);innerIndex++{
      values[outerIndex][innerIndex] = m1.values[outerIndex][innerIndex] * m2.values[outerIndex][innerIndex]
    }
  }
  m.cols = m1.cols
  m.rows = m1.rows
  m.values = values
  return m
}

func vvDot(v1 vector, v2 vector) vector{
  if (v1.length != v2.length){
    log.Fatal("vvDot got vectors of different lengths")
  }
  var nv vector
  nv.length = v1.length
  nv.values = v1.values
  for index, _ := range v1.values{
    nv.values[index] = v1.values[index]*v2.values[index]
  }
  return nv
}

func mmAdd(m1 matrix, m2 matrix) matrix{
  if (m1.rows != m2.rows || m1.cols != m2.cols){
    log.Fatal("cant add matricies of different dimensions")
  }
  var m matrix
  m.rows = m1.rows
  m.cols = m1.cols
  values := make([][]float64, 0)
  for index := range m1.values{
    values = append(values, addArray(m1.values[index], m2.values[index]))
  }
  m.values = values
  return m
}

func smMultiply(scalar float64, m matrix) matrix{
  var nM matrix
  nM.rows = m.rows
  nM.cols = m.cols
  blankRow := make([]float64, nM.cols)
  values := make([][]float64, 0)
  for index:=0;index<nM.rows;index++{
    values = append(values, blankRow)
  }
  for r:=0;r<nM.rows;r++{
    for c:=0;c<nM.cols;c++{
      values[r][c] = m.values[r][c] * scalar
    }
  }
  nM.values = values
  return nM
}

// activation application over a vector
func funcApply(v vector, functionName string) vector{
  if (functionName != "sigmoid" && functionName != "softmax" && functionName != "ReLu"){
    log.Fatal("funcApply tried to apply a function that it does not know")
  }
  if (functionName == "sigmoid"){
    var rv vector
    rv.length = v.length
    values := make([]float64, rv.length)
    for index:=0;index<rv.length;index++{
      values[index] = sigmoid( v.values[index])
    }
    rv.values = values
    return rv
  }
  if (functionName == "softmax") {
    data := softmax(v.values)
    rv := vector{values: data, length: len(data)}
    return rv
  }
  if (functionName == "ReLu"){
    var rv vector
    rv.length = v.length
    values := make([]float64, rv.length)
    for index:=0;index<rv.length;index++{
      values[index] = ReLu( v.values[index])
    }
    rv.values = values
    return rv
  } else {
    log.Fatal("funcApply is dead")
    var v vector
    return v
  }
}

// activations functions
func safeExp(input float64) float64{
  rtn := math.Exp(input)
  if (math.IsNaN(rtn)){
    if (input > 0){
      return float64(999999999999999999)
    } else {
      return float64(0.0000000000000001)
    }
  }
  return rtn
}

func sigmoid(input float64) float64{
  return input
  //return input
  //return (math.Exp(2*input)-1)/(math.Exp(2*input)+1)
  if (math.IsNaN(float64(1)/(float64(1)+ safeExp(-input)))){
    log.Fatal("sigmoid giving Nan values", input)
  }
  return float64(1)/(float64(1)+ safeExp(-input))
}

func ReLu(input float64) float64{
  if (input > 0){
    return input
  }
  return float64(0.000001)
}

func softmax(input []float64) []float64{
  rtn:=make([]float64, 0)
  sum := float64(0)
  for _, el := range input{
    sum = sum + safeExp(el)
    //sum = sum + math.Exp(el)
  }
  for index:=0;index<len(input);index++{
    rtn = append(rtn, safeExp(input[index])/sum)
  }
  if (math.IsNaN(rtn[0])){
    log.Fatal("softmax spitting out nans")
  }
  return rtn
}

// softmax doesnt really seem to work
func softmaxOLD(input []float64) []float64{
  rtn := make([]float64, 0)
  sum := float64(0)
  for _,v := range input{
    sum = sum + safeExp(-v)
  }
  for _, v := range input{
    if (math.IsNaN(safeExp(-v)/sum)){
      fmt.Println("starting")
      fmt.Println(safeExp(-v))
      fmt.Println(sum)
      log.Fatal("softmax giving Nan values", v)
    }
    rtn = append(rtn, safeExp(v)/sum)
  }
  return rtn
}

// these two should never be used and are probably broken
func flattern(network NN) []float64{
  log.Fatal("don't use flattern it is broken")
  rtn := make([]float64, 0)
  for vi:=0;vi<len(network.biasVectors);vi++{
    for ei:=0;ei<len(network.biasVectors[vi].values);ei++{
      rtn = append(rtn, network.biasVectors[vi].values[ei])
    }
  }

  for mi:=0;mi<len(network.weightMatricies);mi++{
    for ri:=0;ri<len(network.weightMatricies[mi].values);ri++{
      for ci:=0;ci<len(network.weightMatricies[mi].values[ri]);ci++{
        rtn = append(rtn, network.weightMatricies[mi].values[ri][ci])
      }
    }
  }
return rtn
}

func unflattern(flatList []float64, layerSizes []int) NN{
  log.Fatal("dont use unflattern it is broken")
  // the first elements of the list are easy, they are the vectors
  vectors := make([]vector, 0)
  layerSizeSum := int(0)

  for _, v := range layerSizes{
    layerSizeSum = layerSizeSum + v
  }

  // lets get the indexes where vectors start. They end at layerSizeSum
  vectorStartIndexes := make([]int, 0)
  for i, _ := range layerSizes{
    newIndex := 0
    for j, _:= range layerSizes{
      if (j < i){
        newIndex += layerSizes[j]
      }
    }
    vectorStartIndexes = append(vectorStartIndexes, newIndex)
  }

  for index:=0; index<len(vectorStartIndexes); index++{
    nVectorValues := flatList[vectorStartIndexes[index]:layerSizeSum]
    if (index != (len(vectorStartIndexes) - 1)){
      nVectorValues = flatList[vectorStartIndexes[index]:vectorStartIndexes[index+1]]
    }

    newVector := vector{values: nVectorValues, length: len(nVectorValues)}
    vectors = append(vectors, newVector)
  }


  // now for the matricies
  matrixIndexes := make([][]int, 0)
  startIndex := layerSizeSum
  for i := range layerSizes{
    if (i != 0){
      newEntry := []int{startIndex, startIndex+layerSizes[i]*layerSizes[i-1]}
      matrixIndexes = append(matrixIndexes, newEntry)
      startIndex = newEntry[1]
    }
  }
  matrixes := make([]matrix, 0)
  // so thats all the indexes that we need generated
  for i := range matrixIndexes{
    start := matrixIndexes[i][0]
    end := matrixIndexes[i][1]
    data := flatList[start:end]
    matrixes = append(matrixes, matrixConstruct(data, layerSizes[i+1], layerSizes[i]))
  }
  actFunctions := make([]string, 0)
  for index:=0;index<len(vectors);index++{
    actFunctions = append(actFunctions, "ReLu")
  }

  rtn := NN{biasVectors:vectors, activationFunctions: actFunctions, weightMatricies: matrixes}
  return rtn
}


func matrixConstruct(data []float64, rows int, cols int) matrix{
  var m matrix
  m.rows = rows
  m.cols = cols
  zeroRow := make([]float64, cols)
  values := make([][]float64, rows)
  for index:=0;index<rows;index++{
    values[index] = zeroRow
  }

  rowCounter := 0
  for index:=0;index<len(data);index++{
    if (index % cols == 0 && index != 0){
      rowCounter = rowCounter + 1
    }
    values[rowCounter][index % cols]=data[index]
  }
  m.values = values
  return m
}

// the main function for evaluating a neural network
func evaluate(network NN, input []float64) []float64 {
  var vInput vector
  vInput.length = len(input)
  vInput.values = input
  //fmt.Println("how it started", vInput.values)
  vInput = vvAdd(vInput, network.biasVectors[0])
  //fmt.Println("here is the next step", vInput)
  //vInput = funcApply(vInput, "sigmoid")

  length := len(network.biasVectors)
  for index:=1;index<length;index++{
    vInput = mvMultiply(network.weightMatricies[index-1], vInput)
    //fmt.Println(network.weightMatricies[index-1])
    //fmt.Println("and the step after that", vInput)
    vInput = vvAdd(vInput, network.biasVectors[index])
    vInput = funcApply(vInput, network.activationFunctions[index-1])

  }

  if (math.IsNaN(vInput.values[3])){
    log.Fatal("evaluate spitting out Nans")
  }
  //fmt.Println("here is evaluates output", vInput.values)
  //fmt.Println("how it ended", vInput.values)
  return vInput.values
}

func main(){
  rand.Seed(time.Now().UTC().UnixNano())
  numcpu := flag.Int("cpu", runtime.NumCPU()-2, "")
  flag.Parse()
  runtime.GOMAXPROCS(*numcpu)
  fmt.Println("Program Starting")
  filePaths := make([]string, 0)
  baseName := "C:/Users/maxim/OneDrive/Desktop/GoWorkspace/src/goForex/EURfile"
  for index:=1;index<7;index++{
    filePaths = append(filePaths, baseName+strconv.Itoa(index)+".txt")
  }

  period := "m"
  trainingPercentage := float64(float64(5.95)/float64(6))
  headSize := 60
  tailSize := 10
  iteration := 1
  predictionPopulation := 10
  testingData, trainingData := getTrainingAndTestingData(filePaths, period, trainingPercentage, headSize, tailSize, iteration)
  initialPopSize := 12*10
  //weightMatrixSize := headSize
  randomNoise := float64(1.5)
  percentRetain := float64(1)
  fmt.Println("done processing data")
  //ntestingData, _ := normalizeCandles(testingData, -1)
  //refinedData := candleStrip(ntestingData)
  layerSizes := []int{headSize, 10, headSize}
  activationFunctions := []string{"ReLu","softmax","ReLu","ReLu"}
  GTrain(initialPopSize, randomNoise, percentRetain, layerSizes, activationFunctions, &testingData, &trainingData, predictionPopulation)

}

func getFitness(output []float64, expectedOutput []float64) float64{
  sum := float64(0)
  for index:=0;index<len(output);index++{
    sum = sum + (output[index] - expectedOutput[index])*(output[index] - expectedOutput[index])
  }
  return float64(1)/sum
}

// nbeed takes a list of two parents and a random noise percent and produces one
// offspring
func nbreed(parents []NN, randomNoise float64) NN{
  var nNN NN
  p1 := parents[0]
  p2 := parents[1]
  //fmt.Println("starting matricies", p1.weightMatricies[0])
  //fmt.Println("starting matricies", p2.weightMatricies[0])
  if (len(parents) > 2){
    log.Fatal("giving nbreed the wrong parents")
  }
  if (len(parents) == 1){
    return parents[0]
  }

  newBiasVectors := make([]vector, 0)
  for bIndex:=0;bIndex<len(p1.biasVectors);bIndex++{
    newBiasVector := vvAdd(p1.biasVectors[bIndex], p2.biasVectors[bIndex])
    newBiasVector = svMultiply(0.5, newBiasVector)
    deviationVector := vvAdd(p1.biasVectors[bIndex], svMultiply(-1.0, p2.biasVectors[bIndex]))
    deviationVector = vvDot(deviationVector, randomVector(deviationVector.length))
    deviationVector = svMultiply(randomNoise, deviationVector)
    newBiasVector = vvAdd(newBiasVector, deviationVector)
    newBiasVectors = append(newBiasVectors, newBiasVector)
  }
  newMatrixes := make([]matrix, 0)
  for mIndex:=0;mIndex<len(p1.weightMatricies);mIndex++{
    newMatrix := mmAdd(p1.weightMatricies[mIndex], p1.weightMatricies[mIndex])
    deviationMatrix := mmAdd(p1.weightMatricies[mIndex], smMultiply(-1.0, p2.weightMatricies[mIndex]))
    deviationMatrix = mmDot(deviationMatrix, randomMatrix(deviationMatrix.rows, deviationMatrix.cols))
    deviationMatrix = smMultiply(randomNoise, deviationMatrix)
    newMatrix = smMultiply(0.5, newMatrix)
    newMatrix = mmAdd(newMatrix, deviationMatrix)
    //fmt.Println("after being added", newMatrix)
    newMatrixes = append(newMatrixes, newMatrix)
  }
  //fmt.Println("here are the new matricies", newMatrixes[0])
  nNN.weightMatricies = newMatrixes
  nNN.biasVectors = newBiasVectors
  nNN.activationFunctions = p1.activationFunctions
  return nNN
}

// this scoring is only for supervised learning where the solution is known
func scoreAgents(agents []NN, input []float64, expectedOutput []float64) []float64 {
  scores := make([]float64, len(agents))

  for index:=0; index <len(agents);index++{
    scores[index] = getFitness(evaluate(agents[index], input), expectedOutput)
  }
  return scores
}
// maybe we should not allow something to breed with itself.

// takes two arrays, scores and agents that are correctly ordered and the number of selections to be made
// and returns the selected number of NN agents
func weightedSelect(scores []float64, agents []NN, num int) []NN{
  //lowestScore, _ := findMin(scores)
  //for index:=0;index<len(scores);index++{
    //scores[index] = scores[index] - lowestScore
  //}
  num = 2 // we always use it like this
  sum := float64(0)
  cumSum := make([]float64, 0)
  for index:=0;index<len(scores);index++{
    sum = sum + scores[index]
    cumSum = append(cumSum, sum)
  }
  cumSum = multiplyArray(float64(1)/sum, cumSum)
    //fmt.Println(cumSum)
  selectedAgents := make([]NN, 0)
  lastIndex := -1
  for index:=0;index<num;index++{
    selectionKey := rand.Float64()
    closestIndex := 0
    minDev := float64(9999)
    for i, el := range cumSum{
      if (el-selectionKey >= 0){
        if (el-selectionKey < minDev){
          closestIndex = i
          minDev = el-selectionKey
        }
      }
    }
    if (lastIndex == -1){
      lastIndex = closestIndex
      selectedAgents = append(selectedAgents, agents[closestIndex])
    } else if (lastIndex != -1 && lastIndex == closestIndex){
      index = index - 1
      fmt.Println("stopped breeding with itself")
    } else {
      selectedAgents = append(selectedAgents, agents[closestIndex])
    }


  }
  return selectedAgents
}

func ngetNextGeneration(currentGeneration []NN, input []float64, expectedOutput []float64, randomNoise float64, percentRetain float64) []NN{
    if (len(currentGeneration) == 1){
      return currentGeneration
    }
    scores := scoreAgents(currentGeneration, input, expectedOutput)
    rtn := make([]NN, 0)
    currentPop := 0
    for (currentPop < int(math.Floor(float64(len(currentGeneration))*percentRetain))){
      selectedPop := weightedSelect(scores, currentGeneration, 2)
      rtn = append(rtn, nbreed(selectedPop, randomNoise))
      currentPop = currentPop + 1
    }
    fmt.Println("here is this generations average score")
    fmt.Println(newArrayAverage(scores))
    fmt.Println("done")
    return rtn
}

func generatePopulation(layerSizes []int, activationFunctions []string, number int) []NN{
  rtn := make([]NN, 0)
  for index:=0;index<number;index++{
    rtn = append(rtn, generateRandomNetwork(layerSizes, activationFunctions))
  }
  return rtn
}
