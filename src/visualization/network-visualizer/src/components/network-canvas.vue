<template>
  <div id="three-canvas-container"></div>
  <div class="hidden-hack">{{props.apiData}}</div>
</template>

<script setup>
import * as THREE from 'three';
import {onMounted, defineProps, onUpdated, watch} from "vue";
import { CSS2DRenderer } from 'three/addons/renderers/CSS2DRenderer.js';

const startingPosition = 3.5;
const initialXOffset = -1.75;

const NEURON_RADIUS = 1;
const NEURON_OFFSET = 2 * NEURON_RADIUS + NEURON_RADIUS / 4;
const NEURON_DETAIL = 16;
const MAX_NEURONS_PER_ROW = 50;

let scene;
let camera;
let renderer;

const props = defineProps(['apiData']);
console.log(props.apiData)

// watch(props.apiData, async (newValue, oldValue) => {
//   await initializeNetworkVisualization()
// });

/**
 * Creates the following ThreeJS objects:
 * - Scene
 * - PerspectiveCamera
 * - Renderer
 *
 * And then appends the canvas to the DOM.
 */
function initializeThreeJS() {
  scene = new THREE.Scene()

  camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.01, 5000);
  camera.position.x = 50;
  camera.position.y = -50;
  camera.position.z = 100;

  renderer = new THREE.WebGLRenderer();
  const container = document.getElementById('three-canvas-container');
  renderer.setSize(container.offsetWidth, container.offsetHeight);
  renderer.setClearColor(0x222222, 1);
  
  renderer.domElement.id = 'three-canvas'
  container.appendChild(renderer.domElement);
}

/**
 * Creates the actual network representation.
 * @returns {Promise<void>}
 */
async function generateNetworkVisualization() {
  const neuron_layers = [];

  if (!props.apiData) {
    scene.remove.apply(scene, scene.children);
    return;
  }

  const layers = props.apiData.layers;

  let offset = 0;
  // const ctx = document.getElementById("three-canvas").getContext("2d");
  // console.log(document.getElementById("three-canvas").getContext("3d"))
  // ctx.font = "48px serif";
  // ctx.fillText("Hello world", 10, 50);
  for (let layer_index in layers) {
    const layer = layers[layer_index];
    const amountOfNeurons = layer.units;

    const layer_xOffset = initialXOffset + offset;
    const numberOfRows = Math.ceil(amountOfNeurons / MAX_NEURONS_PER_ROW);
    neuron_layers.push(generateNeurons(scene, amountOfNeurons, layer_xOffset, layer.outputs));
    offset += layer_index * (8 * NEURON_RADIUS);
    // TODO: Fix this offset, the calculation here is incorrect.
    offset += numberOfRows * 2.5 * NEURON_RADIUS;
  }
}

/**
 * Generates the circles that are used to display neurons.
 *
 * @param scene
 * @param layerSize
 * @param xOffset
 * @param {Array} outputs
 */
function generateNeurons(scene, layerSize, xOffset, outputs) {
  const neurons = [];
  outputs = outputs.map((output) => parseFloat(output));

  const min = Math.min(...outputs)
  const max = Math.max(...outputs);
  const difference = max - min;

  outputs = outputs.map((output) => output - min);
  outputs = outputs.map((output) => 1 / difference * output);


  for (let i = 0; i < layerSize; i++) {
    const output = outputs[i];
    let scaledOutput = 255 * output;
    scaledOutput = Math.round(scaledOutput)

    const color = `rgb(${scaledOutput}, ${scaledOutput}, ${scaledOutput})`;

    const geometry = new THREE.CircleGeometry(NEURON_RADIUS, NEURON_DETAIL);
    const material = new THREE.MeshBasicMaterial({color: color});
    const neuron = new THREE.Mesh(geometry, material);
    if (i % MAX_NEURONS_PER_ROW === 0) {
      xOffset = xOffset + 2 * NEURON_RADIUS;
    }
    neuron.position.y += startingPosition - (i % MAX_NEURONS_PER_ROW) * NEURON_OFFSET;
    neuron.position.x += xOffset
    scene.add(neuron);
    neurons.push(neuron);
  }
  return neurons;
}

function cameraSpeed(zoomSpeed) {
  return 10 * zoomSpeed;
}

function zoomSpeed(zoom) {
  return zoom / 50;
}

document.addEventListener('keydown', (event) => {
  const currentZoomSpeed = zoomSpeed(camera.position.z);

  switch (event.key) {
    case 'ArrowUp':
      camera.position.y += cameraSpeed(currentZoomSpeed);
      break;
    case 'ArrowDown':
      camera.position.y -= cameraSpeed(currentZoomSpeed);
      break;
    case 'ArrowLeft':
      camera.position.x -= cameraSpeed(currentZoomSpeed);
      break;
    case 'ArrowRight':
      camera.position.x += cameraSpeed(currentZoomSpeed);
      break;
  }
});

document.addEventListener('wheel', (event) => {
  if (event.deltaY > 0) {
    camera.position.z += zoomSpeed(camera.position.z);
  } else if (event.deltaY < 0) {
    camera.position.z -= zoomSpeed(camera.position.z);
  }
})


function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}

onMounted(() => {
  initializeThreeJS();
  generateNetworkVisualization();
  animate();
});

onUpdated(() => {
  generateNetworkVisualization();
})
</script>

<style scoped>
#three-canvas-container {
  width: 100vw;
  height: 80vh;
}

/*
TODO: This is a stupid solution to the following problem:
      When this component is initialized, the request to the server has not yet been completed.
      Therefore we need to wait for the props to be updated, however, if we don't somehow use the props, Vue is so
      helpful and just doesn't update the component... There most certainly exists a better solution for this.
 */
.hidden-hack {
  display: none;
}
</style>
