function CircumplexManager(idDiv) {
    this.idObj = "#" + idDiv;
    this.htmlObj = $(this.idObj);
}

CircumplexManager.prototype.plotPoints = function(points, emotions) {
    for (var i = 0; i < points.length; i++) {
        points[i]["emo_name"] = emotions[points[i]["emotion"]]["desc"];
    }

    $(this.idObj + " canvas").remove();

    var visSpec = {
        $schema: vegaSchema,
        description: 'Arousal - Valence model',
        height: this.htmlObj.height(),
        width: this.htmlObj.height(),
        data: {
            values: points
        },
        mark: "circle",
        encoding: {
            x: {
                title: 'Valence',
                field: 'valence', type: 'quantitative',
                scale: {"domain": [1, 9]},
                axis: {"grid": false, "tickCount": 9}

            },
            y: {
                title: 'Arousal',
                field: 'arousal', type: 'quantitative',
                scale: {"domain": [1, 9]},
                axis: {"grid": false, "tickCount": 9}
            },
            color: {
                field: "emo_name",
                type: "nominal",
                legend: {
                    title: 'Emotion'
                }
            },
            size: {
                title: "Num. instances (SXX-vY)",
                aggregate: "count",
                type: "ordinal"
            }
        }
    };
    vegaEmbed(this.idObj, visSpec);
}

CircumplexManager.prototype.plotConfusionMatrix = function(idObj, emotions, dataTrue, dataPred) {
    // Format data
    var mat = Array(emotions.length).fill(null).map(() => Array(emotions.length).fill(0));
    var tot = Array(emotions.length).fill(0);
    for (var i=0; i < dataTrue.length; i++) {
        mat[dataTrue[i]['emotion']][dataPred[i]['emotion']] += 1;
        tot[dataTrue[i]['emotion']] += 1;
    }
    console.log("confusion_matrix", mat, tot);

    var data = [];
    for (var i = 0; i < emotions.length; i++) {
        if (tot[i] > 0) {
            for (var j = 0; j < emotions.length; j++) {
                data.push({
                    'emoTrue': emotions[i]["name"],
                    'emoPred': emotions[j]["name"],
                    'value': mat[i][j] * 1.0 / tot[i]
                });
            }
        }
    }
    console.log(data);

    var visSpec = {
        $schema: vegaSchema,
        //height: this.htmlObj.height() - 150,
        //width: this.htmlObj.width() - 60,
        data: {
            values: data
        },
        "config": {
          "view": {
              "strokeWidth": 0,
              "step": 13
          },
          "axis": {
              "domain": false
          }
        },
        mark: "rect",
        encoding: {
            x: {
                title: null,
                field: "emoPred",
                type: "nominal"
            },
            y: {
                title: null,
                field: "emoTrue",
                type: "nominal"
            },
            color: {
                field: "value",
                type: "quantitative",
                scale: {"domain": [0,1]},
                legend: {
                    title: null,
                    direction: "horizontal",
                    orient: "bottom"
                }
            }
        }
    }
    vegaEmbed(idObj, visSpec);
}