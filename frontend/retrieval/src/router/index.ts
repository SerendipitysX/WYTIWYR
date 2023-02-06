import Vue from "vue";
import VueRouter, { RouteConfig } from "vue-router";
import DashBoard from "../views/DashBoard.vue";

Vue.use(VueRouter);

const routes: Array<RouteConfig> = [
  {
    path: "/",
    name: "home",
    component: DashBoard,
  },
];

const router = new VueRouter({
  mode: "hash",
  base: process.env.BASE_URL,
  routes,
});

export default router;
