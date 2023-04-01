<template>
  <div>
    <div id="metadata-container" v-if="apiData !== null">
      <div class="model-settings">
        <label for="model-name-input">Model name</label>
        <select class="form-control" v-model="modelName">
          <option v-for="model in models" :value="model">{{ model }}</option>
        </select>

        <label for="document-index-input">Input Text</label>
        <input id="document-content-input" v-model="documentContent">
        <button @click="getModel">Load Model</button>
      </div>

      <div v-if="apiData.value !== null">
        <div>Content: {{ apiData['documentTokens'] }}</div>
        <div>Prediction: {{ apiData['prediction'] }}</div>
      </div>

    </div>
    <network-canvas :api-data="apiData"/>
  </div>
</template>

<script setup lang="ts">
import NetworkCanvas from "./components/network-canvas.vue";
import {onMounted, Ref, ref} from "vue";

const apiData: Ref<{} | null> = ref(null);
const models: Ref<[]> = ref([]);
const modelName: Ref<string> = ref('1679669805');
const documentContent: Ref<string> = ref('Hello World');

async function getModelList() {
  let apiResponse;
  apiData.value = null;
  try {
    apiResponse = await fetch('http://localhost:5000/models');
  } catch (error) {
    alert("Could not fetch data from the API! Is it running?");
    return;
  }

  if (!apiResponse || apiResponse.status !== 200) {
    alert("API returned non 200 status code!");
  }

  models.value = await apiResponse.json();
}

async function getModel() {
  let apiResponse;
  apiData.value = null;
  try {
    apiResponse = await fetch(
        'http://localhost:5000', {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            modelName: modelName.value,
            documentContent: documentContent.value,
          }),
        }
    );
  } catch (error) {
    alert("Could not fetch data from the API! Is it running?");
    return;
  }

  if (!apiResponse || apiResponse.status !== 200) {
    alert("API returned non 200 status code!");
  }

  apiData.value = await apiResponse.json();
}


onMounted(async () => {
  getModel();
  getModelList();
})

</script>
<style scoped>

.model-settings {
  display: grid;
  grid-template-columns: 1fr 1fr;
  max-width: 500px;
  margin: 0 0 1rem 0;
}
</style>
