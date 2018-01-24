<template>
  <div class="demo">
    <transition name="fade">
      <model-status v-if="modelLoading || modelInitializing" 
        :modelLoading="modelLoading"
        :modelLoadingProgress="modelLoadingProgress"
        :modelInitializing="modelInitializing"
        :modelInitProgress="modelInitProgress"
      ></model-status>
    </transition>
    <v-alert outline color="error" icon="priority_high" :value="!hasWebGL">
      Note: this browser does not support WebGL 2 or the features necessary to run in GPU mode.
    </v-alert>
    <v-layout row wrap justify-center>
      <v-flex sm6 md4>
        <div class="input-column">
          <div class="input-container">
            <div class="input-label">Draw any digit (0-9) here <span class="arrow">⤸</span></div>
            <img src="/demos/data/test2.jpg" id="image-in" ></img>
          </div>
        </div>
      </v-flex>
      <v-flex sm2 md1>
        <div class="controls-column">
          <div class="control">
            <v-switch
              :disabled="modelLoading || modelInitializing || !hasWebGL"
              label="use GPU"
              v-model="useGPU"
              color="primary"
            ></v-switch>
          </div>
          <div class="control">
            <v-btn 
              :disabled="modelLoading || modelInitializing"
              flat
              bottom
              right
              color="primary"
              @click="deactivateDrawAndPredict"
            >
              <v-icon right>done</v-icon>
              Run
            </v-btn>
            <v-btn 
              :disabled="modelLoading || modelInitializing"
              flat
              bottom
              right
              color="primary"
              @click="clear"
            >
              <v-icon left>close</v-icon>
              Clear
            </v-btn>
          </div>
        </div>
      </v-flex>
    </v-layout>
    </div>
  </div>
</template>

<script>
import _ from 'lodash'
import { mathUtils, tensorUtils } from '../../utils'
import ModelStatus from '../common/ModelStatus'
import ops from 'ndarray-ops'
import ndarray from 'ndarray'
import unpack from 'ndarray-unpack'
import pack from 'ndarray-pack'

window.range = a=> Array.from( new Uint32Array(a) ).map((v,i) => i )
window.uu = unpack
window.imshow = require("ndarray-imshow")
window.concatcols = require('ndarray-concat-cols');
window.tdebug = require('../../tdebug.json')
window.pack = pack;
window.ops = ops;
window.ndarray = ndarray;

tdebug.forEach(function(v){
  v.data = pack( v.data );
});

const MODEL_FILEPATH_PROD = 'https://transcranial.github.io/keras-js-demos-data/mnist_cnn/mnist_cnn.bin'
// const MODEL_FILEPATH_DEV = '/demos/data/mnist_cnn/mnist_cnn.bin'
const MODEL_FILEPATH_DEV = '/demos/data/pottan.bin'

const LAYER_DISPLAY_CONFIG = {
  conv2d_1: { heading: '32 3x3 filters, padding valid, 1x1 strides', scalingFactor: 2 },
  activation_1: { heading: 'ReLU', scalingFactor: 2 },
  conv2d_2: { heading: '32 3x3 filters, padding valid, 1x1 strides', scalingFactor: 2 },
  activation_2: { heading: 'ReLU', scalingFactor: 2 },
  max_pooling2d_1: { heading: '2x2 pooling, 1x1 strides', scalingFactor: 2 },
  dropout_1: { heading: 'p=0.25 (only active during training phase)', scalingFactor: 2 },
  flatten_1: { heading: '', scalingFactor: 2 },
  dense_1: { heading: 'output dimensions 128', scalingFactor: 4 },
  activation_3: { heading: 'ReLU', scalingFactor: 4 },
  dropout_2: { heading: 'p=0.5 (only active during training phase)', scalingFactor: 4 },
  dense_2: { heading: 'output dimensions 10', scalingFactor: 8 },
  activation_4: { heading: 'Softmax', scalingFactor: 8 }
}

export default {
  props: ['hasWebGL'],

  components: { ModelStatus },

  created() {

    // store module on component instance as non-reactive object
    this.model = new KerasJS.Model({
      filepath: process.env.NODE_ENV === 'production' ? MODEL_FILEPATH_PROD : MODEL_FILEPATH_DEV,
      gpu: this.hasWebGL,
      transferLayerOutputs: !true
    })

    this.model.events.on('loadingProgress', this.handleLoadingProgress)
    this.model.events.on('initProgress', this.handleInitProgress)
  },

  async mounted() {
    await this.model.ready()
    await this.$nextTick()

    this.canvas = document.createElement('canvas')
  },

  beforeDestroy() {
    this.model.cleanup()
    this.model.events.removeAllListeners()
  },

  data() {
    return {
      useGPU: this.hasWebGL,
      modelLoading: true,
      modelLoadingProgress: 0,
      modelInitializing: true,
      modelInitProgress: 0,
    }
  },

  watch: {
    useGPU(value) {
      this.model.toggleGPU(value)
    }
  },

  methods: {
    handleLoadingProgress(progress) {
      this.modelLoadingProgress = Math.round(progress)
      if (progress === 100) {
        this.modelLoading = false
      }
    },
    handleInitProgress(progress) {
      this.modelInitProgress = Math.round(progress)
      if (progress === 100) {
        this.modelInitializing = false
      }
    },
    clear() {
      const ctx = document.getElementById('input-canvas').getContext('2d')
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
      const ctxCenterCrop = document.getElementById('input-canvas-centercrop').getContext('2d')
      ctxCenterCrop.clearRect(0, 0, ctxCenterCrop.canvas.width, ctxCenterCrop.canvas.height)
      const ctxScaled = document.getElementById('input-canvas-scaled').getContext('2d')
      ctxScaled.clearRect(0, 0, ctxScaled.canvas.width, ctxScaled.canvas.height)
      this.output = new Float32Array(10)
    },
    deactivateDrawAndPredict: _.debounce(
      function() {
        const img = document.getElementById('image-in');
        const imgH = img.height;
        const imgW = img.width;
        const canvas = this.canvas;
        canvas.height = imgH;
        canvas.width = imgW;
        const ctx = canvas.getContext('2d')
        ctx.drawImage( img, 0, 0);

        // scaled to 32 x 32
        const ctxScaled = ctx
        const ctxCenterCrop = ctx
        ctxScaled.save()
        // ctxScaled.scale(32 / ctxCenterCrop.canvas.width, 32 / ctxCenterCrop.canvas.height)
        const imageDataScaled = ctxScaled.getImageData(0, 0, imgW, imgH )
        ctxScaled.restore()

        // process image data for model input
        let data = ndarray (
          Float32Array.from(imageDataScaled.data),
          [ imageDataScaled.width, imageDataScaled.height, 4 ]
        )
        data = data.pick( null, null, 1 )
        data = ops.subseq( data, 127.5 )
        data = ops.divseq( data, 127.5 )
        this.input = Float32Array.from( pack( unpack( data ) ).data )
        window.pottanRunning = true;

        this.model.predict({ input: this.input }).then(outputData => {
          const nClasses = 136
          const nPredictions = outputData.output.length/nClasses;
          outputData=ndarray( outputData.output,[ nPredictions, nClasses ] );
          const predictions = range(nPredictions).map( i => ops.argmax( outputData.pick(i,null))[0] )
          console.log( predictions );
        })
      },
      200,
      { leading: true, trailing: true }
    ),
  }
}
</script>

<style scoped lang="postcss">
@import '../../variables.css';

.input-column {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;

  & .input-container {
    width: 100%;
    text-align: right;
    margin: 20px;
    position: relative;
    user-select: none;

    & .input-label {
      font-family: var(--font-cursive);
      font-size: 18px;
      color: var(--color-lightgray);
      text-align: right;

      & span.arrow {
        font-size: 36px;
        color: #cccccc;
        position: absolute;
        right: -32px;
        top: 8px;
      }
    }

    & .canvas-container {
      display: inline-flex;
      justify-content: flex-end;
      margin: 10px 0;
      border: 15px solid var(--color-green-lighter);
      transition: border-color 0.2s ease-in;

      &:hover {
        border-color: var(--color-green-light);
      }

      & canvas {
        background: whitesmoke;

        &:hover {
          cursor: crosshair;
        }
      }
    }
  }
}

.controls-column {
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: space-between;
  font-family: var(--font-monospace);
  padding-top: 80px;

  & .control {
    width: 100px;
    margin: 10px 0;
  }
}

.output-column {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-left: 40px;

  & .output {
    height: 160px;
    display: flex;
    flex-direction: row;
    align-items: flex-end;
    justify-content: center;
    user-select: none;
    cursor: default;

    & .output-class {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 0 6px;
      border-bottom: 2px solid var(--color-green-lighter);

      & .output-label {
        font-family: var(--font-monospace);
        font-size: 1.5rem;
        color: var(--color-lightgray);
      }

      & .output-bar {
        width: 8px;
        background: #eeeeee;
        transition: height 0.2s ease-out;
      }
    }

    & .output-class.predicted {
      border-bottom-color: var(--color-green);

      & .output-label {
        color: var(--color-green);
      }
    }
  }
}

.layer-outputs-container {
  position: relative;

  & .bg-line {
    position: absolute;
    z-index: 0;
    top: 0;
    left: 50%;
    background: whitesmoke;
    width: 15px;
    height: 100%;
  }

  & .layer-output {
    position: relative;
    z-index: 1;
    margin: 30px 20px;
    background: whitesmoke;
    border-radius: 10px;
    padding: 20px;
    overflow-x: auto;

    & .layer-output-heading {
      font-size: 1rem;
      color: #999999;
      margin-bottom: 10px;
      display: flex;
      flex-direction: column;
      font-size: 12px;

      & span.layer-class {
        color: var(--color-green);
        font-size: 14px;
        font-weight: bold;
      }
    }

    & .layer-output-canvas-container {
      display: inline-flex;
      flex-wrap: wrap;
      background: whitesmoke;

      & canvas {
        border: 1px solid lightgray;
        margin: 1px;
      }
    }
  }
}

/* vue transition `fade` */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.5s;
}
.fade-enter,
.fade-leave-to {
  opacity: 0;
}
</style>
