(function() {
  // 1. 实例化对象
  var myChart = echarts.init(document.querySelector(".pie1  .chart"));
  // 2. 指定配置项和数据
  var option = {
     legend: {
       top: "100%",
       itemWidth: 10,
       itemHeight: 10,
       textStyle: {
         color: "rgba(255,255,255,.5)",
         fontSize: "12"
       }
     },

    tooltip: {
      trigger: "item",
      formatter: "{a} <br/>{b} : {c} ({d}%)"
    },
    // 注意颜色写的位置
    color: [
      "#006cff",
      "#60cda0",
      "#ed8884",
      "#ff9f7f",
      "#0096ff",
      "#9fe6b8",
      "#32c5e9",
      "#1d9dff"
    ],
    series: [
      {
        name: "Quantity statistics",
        type: "pie",
        // 如果radius是百分比则必须加引号
        radius: ["20%", "50%"],
        center: ["50%", "45%"],
        roseType: "radius",
        data: [
          { value: 29, name: "Hammondia hammondi" },
          { value: 172, name: "Toxoplasma" },
          { value: 16, name: "Neospora" },
          { value: 9, name: "Cystoisospora suis" },
          { value: 19, name: "Plasmodium falciparum" },
        ],
        // 修饰饼形图文字相关的样式 label对象
        label: {
          fontSize: 11
        },
        // 修饰引导线样式
        labelLine: {
          // 连接到图形的线长度
          length: 8,
          // 连接到文字的线长度
          length2: 8,
        }
      }
    ]
  };

  // 3. 配置项和数据给我们的实例化对象
  myChart.setOption(option);
  // 4. 当我们浏览器缩放的时候，图表也等比例缩放
  window.addEventListener("resize", function() {
    // 让我们的图表调用 resize这个方法
    myChart.resize();
  });
})();