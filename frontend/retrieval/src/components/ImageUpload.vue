<template>
  <div style="display: flex; justify-content: start; flex-direction: column">
    <div
      class="title_bold"
      style="width: -webkit-fill-available;height: -webkit-fill-available text-align: start"
    >
      Annotation
    </div>
    <div style="width: 100%; padding: 10px">
      <v-img
        :src="
          fileUrl ??
          'https://retrieval-1304458321.cos.ap-nanjing.myqcloud.com/all/barchart_137.png'
        "
        style="width: 100%; height: 25vh"
        contain
      >
      </v-img>
      <div
        style="
          flex-direction: row;
          width: -webkit-fill-available;

          padding: 10px;
        "
      >
        <v-file-input
          style="width: 25vw; margin-left: 5vw"
          @click="hasMoreImg = true"
          :clearable="false"
          v-model="imgFile"
          dense
          chips
          @change="
            Annotation();
            showAddImg = true;
          "
          :aspect-ratio="1"
          outlined
          placeholder="click to upload an image"
          accept="image/* "
          color="#bbb"
        >
        </v-file-input>
      </div>
      <v-skeleton-loader
        v-if="$store.state.loading.annotation"
        style="width: 300px"
        type="list-item-three-line,list-item-three-line,list-item-three-line"
      ></v-skeleton-loader>
      <div
        v-if="!$store.state.loading.annotation"
        style="display: flex; height: 100%; width: 100%; margin-top: -15px"
        class="d-flex flex-column"
      >
        <v-divider></v-divider>
        <div class="hint mt-2">Explicite attributes</div>
        <div
          v-for="attribute in Object.keys(
            $store.state.annotation_returned ?? {}
          )"
          :key="attribute"
          class="mb-2 title"
        >
          {{ attribute }}
          <div>
            <span
              style="
                border: 0.5px solid #dee2e6;
                border-radius: 5px;
                padding: 5px;
                margin-right: 5px;
              "
              v-for="chip in $store.state.annotation_returned[attribute]"
              :key="chip"
              class="attributes"
            >
              {{ chip }}
            </span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
<script>
export default {
  watch: {
    imgFile(val, oldVal) {
      if (oldVal == null && val != null) this.closeCropper = false;
      else if (oldVal != val && this.showAddImg == true) {
        this.closeCropper = false;
      }
      if (!this.closeCropper) this.showAddImg = true;
      else this.showAddImg = false;
      if (val == undefined) {
        this.closeCropper = false;
        this.fileUrl = null;
      }
    },
  },
  data() {
    return {
      showAddImg: false,
      imgFile: null,
      fileUrl: undefined,
      hasImgUpload: 0,
      snackbarColor: "#bbb",
    };
  },
  methods: {
    zoomPreview() {
      if (this.showZoom == 0) {
        this.showZoom = 1;
      } else {
        this.showZoom = 0;
      }
    },
    base64ToFile(urlData, fileName) {
      let arr = urlData.split(",");
      let mime = arr[0].match(/:(.*?);/)[1];

      let bytes = atob(arr[1]);
      let n = bytes.length;
      let ia = new Uint8Array(n);
      while (n--) {
        ia[n] = bytes.charCodeAt(n);
      }
      return new File([ia], fileName, { type: mime });
    },
    retrieval() {
      this.$store.state.loading.retrieval = true;
      this.axios({
        method: "post",
        url: this.$store.state.retrieval_host + "retrieval",
        data: {
          retrieval: [],
          new_clisifier: [],
          user: [],
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
    init_retrieval() {
      this.$store.state.query = {
        default: [],
        requisite: [],
        user_intent: null,
      };
      this.$store.state.selected = {
        default: {
          Type: [],
          Color: [],
          Layout: [],
          Trend: [],
        },
        requisite: {},
      };
    },
    Annotation() {
      if (this.imgFile != null) {
        this.hasImgUpload = 1;
      }
      this.fileUrl = URL.createObjectURL(this.imgFile);
      this.upload();
    },
    async upload() {
      var originalFile = new FormData();
      originalFile.append("file", this.imgFile);
      this.$store.state.uploadImg = originalFile;
      this.$store.state.loading.annotation = true;
      await this.axios({
        method: "post",
        url: this.$store.state.annotation_host + "attr_requisite",
        data: originalFile,
      })
        .then((res) => {
          this.init_retrieval();
          this.$store.state.loading.annotation = false;
          this.$store.state.annotation_returned = JSON.parse(
            JSON.stringify(res.data)
          );
          if (this.$store.state.annotation_returned["Color"] == "Single Color")
            this.$store.state.annotation_returned["Color"] = [
              "Categorical Colormap",
            ];

          this.axios({
            method: "post",
            url: this.$store.state.annotation_host + "attr_intent",
            data: {
              list: this.$store.state.newclassifier,
            },
          })
            .then((res) => {
              this.$store.state.clip = res.data["CLIP"];
              this.retrieval();
            })
            .catch((error) => {
              this.$store.state.showError = true;
              console.log(error);
            });
        })
        .catch((error) => {
          this.$store.state.loading.annotation = false;
          this.$store.state.showError = true;
          console.log(error);
        });
    },
  },
};
</script>
<style>
.view {
  margin: 20px;
  padding: 10px 20px 10px 20px;
  border-radius: 17px;
  border: 1px solid #aaa;
  width: 50vw;
}
.attributes {
  font-size: 16px;
  color: #aaa;
}
</style>
