(function() {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var accent3 = style.getPropertyValue('--accent3').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var bg2 = style.getPropertyValue('--bg2').trim();

  // ===== Radar Chart: RAG Quality =====
  var radarEl = document.getElementById('chart-radar');
  if (radarEl) {
    var radar = echarts.init(radarEl, null, { renderer: 'svg' });
    radar.setOption({
      animation: false,
      tooltip: {
        trigger: 'item',
        backgroundColor: bg2,
        borderColor: rule,
        textStyle: { color: ink, fontFamily: 'Outfit' }
      },
      radar: {
        indicator: [
          { name: 'Recall@3', max: 1.0 },
          { name: 'Precision@3', max: 1.0 },
          { name: 'MRR', max: 1.0 },
          { name: 'Avg Similarity', max: 1.0 },
          { name: '稳定性', max: 1.0 }
        ],
        shape: 'polygon',
        splitNumber: 4,
        axisName: {
          color: ink,
          fontSize: 13,
          fontFamily: 'Outfit',
          fontWeight: 600
        },
        splitLine: { lineStyle: { color: rule } },
        splitArea: { areaStyle: { color: ['rgba(255,107,107,0.02)', 'rgba(255,107,107,0.05)'] } },
        axisLine: { lineStyle: { color: rule } }
      },
      series: [{
        type: 'radar',
        data: [{
          value: [1.0, 0.75, 0.958, 0.48, 0.92],
          name: 'LoveMender RAG',
          areaStyle: { color: 'rgba(255,107,107,0.2)' },
          lineStyle: { color: accent, width: 2 },
          itemStyle: { color: accent },
          symbolSize: 6
        }]
      }]
    });
    window.addEventListener('resize', function() { radar.resize(); });
  }

  // ===== Bar Chart: MRR per Test Case =====
  var mrrEl = document.getElementById('chart-mrr');
  if (mrrEl) {
    var mrr = echarts.init(mrrEl, null, { renderer: 'svg' });
    var cases = [
      '愤怒情绪', '焦虑情绪', '情绪管理', '心情低落',
      '人际冲突', '抑郁倾向', '情绪低谷', '放松方法',
      '情绪技巧', '颗粒度提升', '表达性书写', '情绪需求'
    ];
    var mrrValues = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0];
    var colors = mrrValues.map(function(v) {
      return v >= 1.0 ? accent : accent2;
    });

    mrr.setOption({
      animation: false,
      tooltip: {
        trigger: 'axis',
        backgroundColor: bg2,
        borderColor: rule,
        textStyle: { color: ink, fontFamily: 'Outfit' },
        formatter: function(params) {
          return params[0].name + '<br/>MRR: <strong>' + params[0].value + '</strong>';
        }
      },
      grid: { left: '3%', right: '4%', bottom: '15%', top: '5%', containLabel: true },
      xAxis: {
        type: 'category',
        data: cases,
        axisLabel: {
          color: muted,
          fontSize: 10,
          fontFamily: 'Outfit',
          rotate: 35,
          interval: 0
        },
        axisLine: { lineStyle: { color: rule } },
        axisTick: { show: false }
      },
      yAxis: {
        type: 'value',
        min: 0,
        max: 1.0,
        axisLabel: {
          color: muted,
          fontFamily: 'JetBrains Mono',
          fontSize: 11
        },
        splitLine: { lineStyle: { color: rule, type: 'dashed' } },
        axisLine: { show: false }
      },
      series: [{
        type: 'bar',
        data: mrrValues.map(function(v, i) {
          return { value: v, itemStyle: { color: colors[i], borderRadius: [4, 4, 0, 0] } };
        }),
        barWidth: '55%',
        label: {
          show: true,
          position: 'top',
          color: ink,
          fontSize: 9,
          fontFamily: 'JetBrains Mono',
          formatter: function(params) {
            return params.value.toFixed(2);
          }
        }
      }]
    });
    window.addEventListener('resize', function() { mrr.resize(); });
  }
})();
