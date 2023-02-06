<template lang="">
  <div
    class="main"
    style="align-items: center; height: calc(100vh - 100px); padding: 0 10px"
  >
    <div style="display: flex; flex-direction: row; height: calc(75vh - 115px)">
      <ImageUploadVue
        class="view mx-auto"
        style="margin-right: 5px !important; width: 40vw"
      ></ImageUploadVue>
      <AttributeListVue
        class="view mx-auto"
        style="margin-left: 5px !important; overflow-y: scroll; width: 60vw"
      ></AttributeListVue>
    </div>
    <ResultListVue
      class="view"
      style="
        height: calc(25vh + 50px);
        width: calc(100vw - 20px);
        margin-top: -10px;
      "
    ></ResultListVue>
    <v-snackbar
      v-model="$store.state.showError"
      absolute
      bottom
      right
      tile
      color="red accent-2"
    >
      The server is disconnected
    </v-snackbar>
  </div>
</template>
<script>
import ImageUploadVue from "../components/ImageUpload";
import AttributeListVue from "../components/AttributeList.vue";
import ResultListVue from "../components/ResultList.vue";
export default {
  components: {
    ImageUploadVue,
    AttributeListVue,
    ResultListVue,
  },
};
</script>
<style>
.view {
  align-items: center;
  border-radius: 17px;
  border: 0.5px solid #dee2e6;
}
.title_bold {
  font-size: 24px;
  font-weight: 800;
}
html {
  overflow-y: auto;
}
div {
  font-family: Verdana, Geneva, Tahoma, sans-serif;
}
.main {
}
</style>
