<template>
  <div>
    <v-btn
      icon
      outlined
      style="display: flex; position: absolute; right: 30px; top: 30px"
      @click="showDialog = true"
      ><v-icon dark size="30"> mdi-plus </v-icon></v-btn
    >
    <v-dialog v-model="showDialog" persistent width="400px">
      <v-card
        style="width: 100%; height: 50vh; padding: 50px"
        class="d-flex align-center flex-column"
      >
        <div class="d-flex align-start flex-column" style="width: 100%">
          <div class="title">Add a Classifier</div>
          <v-text-field
            style="width: max-content"
            v-model="title"
            label="Classifier Name"
            outlined
            class="mt-6"
            hide-details="auto"
          ></v-text-field>
          <v-divider></v-divider>

          <div class="d-flex align-top mt-6">
            <v-text-field
              outlined
              dense
              v-model="input"
              placeholder="input one choice"
            >
            </v-text-field>
            <v-btn
              color="success"
              style="margin-top: 2px; margin-left: 20px"
              depressed
              @click="add_to_list"
              >Add</v-btn
            >
          </div>
          <v-chip-group column multiple class="">
            <v-chip
              outlined
              close
              v-for="chip in attributes"
              :key="chip"
              @click:close="del(chip)"
            >
              {{ chip }}</v-chip
            >
          </v-chip-group>
        </div>

        <div class="d-flex justify-space-between mt-auto" style="width: 70%">
          <v-btn @click="showDialog = false" depressed>Close</v-btn>
          <v-btn depressed @click="submit">Add</v-btn>
        </div>
      </v-card>
    </v-dialog>
  </div>
</template>
<script>
export default {
  data() {
    return {
      showDialog: false,
      attributes: [],
      input: "",
      title: "",
    };
  },
  props: ["newData"],
  methods: {
    open() {
      this.showDialog = true;
    },
    close() {
      this.showDialog = false;
    },
    add_to_list() {
      this.attributes.push(this.input);
      this.input = "";
    },
    del(chip) {
      let index = this.attributes.indexOf(chip);
      this.attributes.splice(index, 1);
    },

    submit() {
      this.$emit("newData", {
        title: this.title,
        attributes: this.attributes,
      });
      this.input = "";

      this.axios({
        method: "post",
        url: this.$store.state.annotation_host + "attr_intent",
        data: {
          list: this.attributes,
        },
      })
        .then((res) => {
          this.$store.state.newclassifier = this.attributes;
          this.$store.state.newclassifierTitle = this.title;
          this.$store.state.clip = res.data["CLIP"];
          this.attributes = [];
          this.showDialog = false;
        })
        .catch((error) => {
          this.showDialog = false;
          this.$store.state.showError = true;
        });
    },
  },
};
</script>
<style></style>
