const classifier = knnClassifier.create()
const webcamElement = document.getElementById("my_camera")
const button = document.getElementById('submit_button')
const input = document.getElementById('image_url')
const img1 = document.getElementById('img1');
const result = document.getElementById('prediction');

let net
let model

app()

async function app() {

  net = await mobilenet.load()

  const webcam = await tf.data.webcam(webcamElement)

  const addExample = async (elemntId) => {
    img = await webcam.capture()
    console.log(img)
    const activation = net.infer(img, true)
    classifier.addExample(activation, elemntId)
    console.log(classifier)
    img.dispose()
  }

  document.getElementById("animal").addEventListener("click", () => {addExample(0) 
    console.log('Animal Added') })
  document.getElementById("human").addEventListener("click", () =>  {addExample(1)
    console.log('Human Added') })
  document.getElementById("building").addEventListener("click", () =>  {addExample(2)
    console.log('Building Added')})

    button.onclick = () => {
        const url = input.value;
        img1.src = url;
        result.innerText = "Wait For It Guys ....";
    }
    img1.onload = () => {
        doPrediction();
    }

    while (true) {
        if (classifier.getNumClasses() > 0) {
            console.log('classifier', classifier) 
          const img = await webcam.capture()
          const activation = net.infer(img, "conv_preds")
          const result = await classifier.predictClass(activation)
          const classes = ["Animal", "Human", "Building"]
          document.getElementById("console").innerText = `
                    Prediction-->: ${classes[0]}\n
                    Probabilty-->: ${result.confidences[0]}\n
                    Prediction-->: ${classes[1]}\n
                    Probabilty-->: ${result.confidences[1]}\n
                    Prediction-->: ${classes[2]}\n
                    Probabilty-->: ${result.confidences[2]}
                `
    
          img.dispose()
        }
    
        await tf.nextFrame()
      }
    
    function doPrediction() {
        if( model ) {
            model.classify(img1).then(predictions => {
                showPrediction(predictions);
            });
        } else {
            mobilenet.load().then(_model => {
                model = _model;
                model.classify(img1).then(predictions => {
                    showPrediction(predictions);
                });
            });
        }
    }
    
    function showPrediction(predictions) {
        result.innerText = "This might be a " + predictions[0].className;
    }

}



