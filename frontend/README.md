## Frontend

### Environment Setup

1.  Set up Node.js and  environment. Please refer to [here](https://www.digitalocean.com/community/tutorials/node-js-environment-setup-node-js-installation)
2.  Set up Vue.js environment:  `npm install vue`

### Host setting

In file `WYTIWYR\frontend\retrieval\src\store\index.ts`, please set the host that run your backend

```js
import Vue from "vue";
import Vuex from "vuex";

Vue.use(Vuex);

export default new Vuex.Store({
  state: {
    watchlst: ["all_attributes.Type"],
    annotation_host: "http://10.30.11.33:7779/", // change to your own hosts
    retrieval_host: "http://10.30.11.33:7780/",
...
```

### Project setup

1. Go to `WYTIWYR\frontend\retrieval` path

2. Install all the needed packages through npm

    ```
    npm install
    ```

3. Compiles and hot-reloads for development

    ```
    npm run serve
    ```

#### Compiles and minifies for production

```
npm run build
```

