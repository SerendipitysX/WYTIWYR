<template>
  <div class="mx-auto">
    <div class="title_bold">Result</div>
    <div class="d-flex justify-center" v-if="$store.state.loading.retrieval">
      <v-skeleton-loader
        style="margin: 5px; margin-top: 40px"
        width="10vh"
        height="10vh"
        v-for="n in 10"
        :key="n"
        type="image"
      ></v-skeleton-loader>
    </div>

    <div
      v-if="!$store.state.loading.retrieval"
      class="d-flex flex-row flex-wrap overflow-auto"
      style="height: 90%; padding: 5px"
    >
      <v-img
        v-for="n in $store.state.imgList"
        :key="n"
        contain
        style="margin: 5px; border: 0.5px solid #dee2e6"
        height="20vh"
        width="20vh"
        :src="`https://retrieval-1304458321.cos.ap-nanjing.myqcloud.com/all/${n}`"
      ></v-img>
    </div>
  </div>
</template>
<script>
export default {
  data() {
    return {};
  },
};
</script>
<style lang=""></style>
