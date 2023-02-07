<template>
  <v-app>
    <v-app-bar app> Annotation and Retrieval </v-app-bar>
    <v-main>
      <DashBoard></DashBoard>
    </v-main>
  </v-app>
</template>

<script>
import DashBoard from "./views/DashBoard.vue";

export default {
  name: "App",

  components: {
    DashBoard,
  },
  methods: {},

  data() {
    return {};
  },
  created() {
    //
  },
};
</script>
