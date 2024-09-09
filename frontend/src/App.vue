<template>
  <v-container fluid>
    <v-row>
      <v-col cols="12" md="6" v-for="(item, index) in jsonData" :key="index">
        <v-card>
          <v-card-text>{{ item.name }}</v-card-text>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
import jsonData from '@/assets/data.json'; // Beispiel Pfad

export default {
  data() {
    return {
      jsonData
    };
  }
};
</script>

