<template lang="">
  <div class="mx-auto d-flex flex-column" :key="refresh">
    <div style="width: 100%">
      <AddClassifier @newData="addClassifier"></AddClassifier>
      <div class="title_bold">Retrieval</div>
      <v-skeleton-loader
        v-if="$store.state.loading.annotation"
        type="list-item-three-line,list-item-three-line,list-item-three-line,divider,list-item-three-line,list-item-three-line,list-item-three-line,actions"
      ></v-skeleton-loader>
      <!-- annotated visual attributes by defalut -->
      <div v-if="!$store.state.loading.annotation">
        <div
          v-for="attribute in Object.keys($store.state.all_setting)"
          :key="attribute"
          class="mb-2 title"
        >
          {{ attribute }}
          <v-chip-group
            column
            multiple
            v-model="$store.state.selected.default[attribute]"
          >
            <div v-for="(chip, index) in attrlst_defult[attribute]" :key="chip">
              <v-chip filter outlined :color="index == 0 ? 'success' : 'grey'">
                {{ chip }}
              </v-chip>
            </div>
          </v-chip-group>
        </div>
        <!-- new classifier add by user -->
        <div
          v-for="attribute in Object.keys(attrlst_intent)"
          :key="attribute"
          class="mb-2 title"
        >
          {{ attribute }}
          <v-chip-group
            column
            multiple
            v-model="$store.state.selected.requisite[attribute]"
          >
            <div v-for="chip in attrlst_intent[attribute]" :key="chip">
              <v-chip
                filter
                outlined
                :color="chip == $store.state.clip ? 'success' : 'grey'"
              >
                {{ chip }}
              </v-chip>
            </div>
          </v-chip-group>
        </div>
      </div>

      <!-- Retrieval part -->
      <div v-if="!$store.state.loading.annotation">
        <v-divider></v-divider>

        <!-- select attributes in default -->
        <div class="title" v-if="$store.state.query.default.length != 0">
          User Selected Attribute
        </div>

        <v-chip-group column multiple active-class="primary--text">
          <draggable
            v-model="$store.state.query.default"
            chosen-class="chosen"
            group="people"
            animation="200"
            @start="onStart"
            @end="onEnd"
          >
            <v-chip
              outlined
              draggable
              v-for="(attr, key) in $store.state.query.default"
              :key="key"
            >
              {{ attr }}</v-chip
            >
          </draggable>
        </v-chip-group>
        <!-- select attributes in the added classifier -->
        <div class="title" v-if="$store.state.query.requisite.length != 0">
          Added Attribute
        </div>
        <v-chip-group column multiple active-class="primary--text">
          <draggable
            v-model="$store.state.query.requisite"
            chosen-class="chosen"
            group="people"
            animation="200"
            @start="onStart"
            @end="onEnd"
          >
            <v-chip
              outlined
              draggable
              v-for="(attr, key) in $store.state.query.requisite"
              :key="key"
            >
              {{ attr }}</v-chip
            >
          </draggable>
        </v-chip-group>
        <!-- short sentences that show the user intent -->
        <div class="title">User intent</div>
        <v-text-field
          style="margin-top: 5px"
          outlined
          color="#bbb"
          placeholder="Describe the chart you'd like to find!"
          clearable
          v-model="$store.state.query.user_intent"
        ></v-text-field>
      </div>
    </div>
    <div
      class="d-flex flex-column mt-auto"
      v-if="!$store.state.loading.annotation"
    >
      <v-btn class="" @click="retrieval" depressed>Retrieval</v-btn>
    </div>
  </div>
</template>

<script>
import draggable from "vuedraggable";
import AddClassifier from "./AddClassifier.vue";

export default {
  components: {
    draggable,
    AddClassifier,
  },

  data() {
    return {
      refresh: true,

      search: {
        default: {
          Type: [],
          Color: [],
          Layout: [],
          Trend: [],
        },
      },
      show: {
        add_retrieval: true,
      },
      newClassifier: {
        title: "",
        attributes: [],
      },
      user_intent: "",

      drag: false,
      default_setting: {
        Type: ["Barchart"],
        Color: ["Sequential Colormap"],
        Layout: ["Horizontal Layout"],
        Trend: [],
      },
    };
  },
  computed: {
    selected_default_cmp() {
      return JSON.stringify(this.$store.state.selected.default);
    },
    selected_requisite_cmp() {
      return JSON.stringify(this.$store.state.selected.requisite);
    },
    attrlst_defult() {
      if (this.$store.state.annotation_returned != undefined)
        return this.merge(this.$store.state.annotation_returned);
      else return this.merge(this.default_setting);
    },
    attrlst_requisite() {
      return {};
    },
    attrlst_intent() {
      return {};
    },
  },

  watch: {
    selected_default_cmp: {
      handler(val, old_val) {
        this.select_handler(val, old_val, this.attrlst_defult, "default");
      },
      deep: true,
    },
    selected_requisite_cmp: {
      handler(val, old_val) {
        this.select_handler(val, old_val, this.attrlst_intent, "requisite");
      },
      deep: true,
    },
  },
  methods: {
    select_handler(val, old_val, attrlst, type) {
      let val_obj = JSON.parse(val);
      let old_val_obj = JSON.parse(old_val);
      if (Object.keys(val_obj).length == Object.keys(old_val_obj).length)
        for (let key of Object.keys(val_obj)) {
          if ([...val_obj[key]].length > [...old_val_obj[key]].length) {
            this.change_attr(key, attrlst, type);
          } else if ([...val_obj[key]].length < [...old_val_obj[key]].length)
            this.remove(
              [...val_obj[key]],
              [...old_val_obj[key]],
              key,
              type,
              attrlst
            );
        }
    },
    change_attr(attr, attrlst, type) {
      this.search[attr] = [];

      for (let i of [...this.$store.state.selected[type][attr]]) {
        if (!this.search[attr].includes(attrlst[attr][i]))
          this.search[attr].push(attrlst[attr][i]);
      }
      this.$store.state.query[type] = [
        ...new Set(this.$store.state.query[type].concat(this.search[attr])),
      ];
    },
    merge(l) {
      let list = JSON.parse(JSON.stringify(l));
      let merged = JSON.stringify(this.$store.state.all_setting);
      merged = JSON.parse(merged);
      for (let item in list) {
        for (let i = 0; i < list[item].length; i++) {
          if (merged[item].includes(list[item][i])) {
            let index = merged[item].indexOf(list[item][i]);
            merged[item].splice(index, 1);
          }
        }
        if (item == "Type") {
          let lst = [
            "Bar",
            "Circle",
            "Diagram",
            "Distribution",
            "Grid",
            "Line",
            "Map",
            "Point",
            "Text",
            "Tree",
          ];
          if (lst.includes(list[item][0])) list[item].splice(0, 1);
        }
        list[item] = list[item].concat(merged[item]);
      }

      return list;
    },
    remove(val, old, attr, type, attrlst) {
      let chip;
      for (let item of old) {
        if (!val.includes(item)) {
          chip = attrlst[attr][item];
          break;
        }
      }
      let index = this.$store.state.query[type].indexOf(chip);
      this.$store.state.query[type].splice(index, 1);
    },
    addRetrieval() {
      if (!this.$store.state.query.user_intent.includes(this.user_intent)) {
        this.$store.state.query.user_intent.push(this.user_intent);
      }
      this.user_intent = "";
    },

    onStart() {
      this.drag = true;
    },
    onEnd() {
      this.drag = false;
    },

    addClassifier(data) {
      this.attrlst_intent[data.title] = data.attributes;
      this.$store.state.selected.requisite[data.title] = [];
      // deep copy to triger the computed
      let tmp = { ...this.$store.state.selected.requisite };
      this.$store.state.selected.requisite = { ...tmp };
      this.refresh = !this.refresh;
    },

    retrieval() {
      this.$store.state.loading.retrieval = true;
      this.axios({
        method: "post",
        url: this.$store.state.retrieval_host + "retrieval",
        data: {
          retrieval: this.$store.state.query.default,
          new_clisifier: this.$store.state.query.requisite,
          user:
            this.$store.state.query.user_intent == null
              ? []
              : [this.$store.state.query.user_intent],
        },
      })
        .then((res) => {
          this.$store.state.imgList = res.data;
          this.$store.state.loading.retrieval = false;
        })
        .catch(() => {
          this.$store.state.showError = true;
          this.$store.state.loading.retrieval = false;
        });
    },
  },
};
</script>
<style>
.v {
  margin: 20px;
  padding: 20px;
  width: 50vw;
}

.chosen {
  border: solid 2px #4433 !important;
}
input {
  padding: 0px !important;
}
.hint {
  font-size: 14px;
  font-style: italic;
  color: #bbb;
}
</style>
