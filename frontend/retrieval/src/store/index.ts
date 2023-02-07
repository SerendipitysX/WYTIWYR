import Vue from "vue";
import Vuex from "vuex";

Vue.use(Vuex);

export default new Vuex.Store({
  state: {
    watchlst: ["all_attributes.Type"],
    annotation_host: "http://10.30.11.33:7779/",
    retrieval_host: "http://10.30.11.33:7780/",
    annotation_returned: {
      Type: ["Barchart"],
      Trend: ["Increase Trend"],
      Color: ["Sequential Colormap"],
      Layout: ["Horizontal Layout"],
    },
    annotation_intent_returned: undefined,
    newclassifier: [],
    newclassifierTitle: "",
    imgList: [
      "barchart_164.png",
      "barchart_127.png",
      "barchart_129.png",
      "barchart_125.png",
      "barchart_132.png",
    ],
    uploadImg: undefined,
    loading: {
      annotation: false,
      retrieval: false,
    },
    selected: {
      default: {
        Type: [],
        Color: [],
        Layout: [],
        Trend: [],
      },
      requisite: {},
    },
    all_setting: {
      Type: [
        "Barchart",
        "Histogram",
        "Stacked Bar Chart",
        "Box Plot",
        "Circular Bar chart",
        "Scatter Chart",
        "Pie Chart",
        "Circular Packing Chart",
        "Heatmap",
        "Choropleth Map",
        "Line Chart",
        "Dendrogram Chart",
        "Network",
        "Star Plot",
        "Word Cloud",
        "Sankey Diagram",
        "Timeline",
        "Donut Chart",
      ],
      Color: [
        "Sequential Colormap",
        "Diverging Colormap",
        "Categorical Colormap",
      ],
      Trend: ["Increase Trend", "Decrease Trend", "Distribution"],
      Layout: ["Horizontal Layout", "Vertical Layout"],
    },
    showError: false,
    clip: "",
    query: {
      default: [],
      requisite: [],
      user_intent: null,
    },
  },
  getters: {},
  mutations: {},
  actions: {},
  modules: {},
});
