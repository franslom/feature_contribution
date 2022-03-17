var featureManager = new FeatureManager("heatmap");
var circumplexMan = new CircumplexManager("circumplex");
var vegaSchema = 'https://vega.github.io/schema/vega-lite/v4.json';

function loadSignals(data, htmlObj) {
    fetch('load_channels', {
        method: 'POST',
        body: JSON.stringify(data), // data can be `string` or {object}!
        headers:{ 'Content-Type': 'application/json' }
    }).then(res => res.json())
    .catch(error => console.error('Error:', error))
    .then(function(response) {
        if (response) {
            console.log("Response: ", response)
            sampleSize.value = response["sampleSize"];
            var channels = response["channels"];
            for (var i in channels) {
                htmlObj.append("<option value='" + channels[i]['id'] + "'>" + channels[i]['label'] + "</option>");
            }
        }
    });
}

function showMetrics(data) {
    $("#time_exec").text(Math.round(data["duration"] * 100) / 100);
    $("#aro_prec").text(Math.round(data["res_aro"]["prec_mean"] * 100) / 100);
    $("#aro_rec").text(Math.round(data["res_aro"]["rec_mean"] * 100) / 100);
    $("#aro_acc").text(Math.round(data["res_aro"]["acc_mean"] * 100) / 100);
    $("#aro_f1").text(Math.round(data["res_aro"]["f1_mean"] * 100) / 100);
    $("#val_prec").text(Math.round(data["res_val"]["prec_mean"] * 100) / 100);
    $("#val_rec").text(Math.round(data["res_val"]["rec_mean"] * 100) / 100);
    $("#val_f1").text(Math.round(data["res_val"]["f1_mean"] * 100) / 100);
    $("#val_acc").text(Math.round(data["res_val"]["acc_mean"] * 100) / 100);
    $("#quad_prec").text(Math.round(data["res_quad"]["prec_mean"] * 100) / 100);
    $("#quad_rec").text(Math.round(data["res_quad"]["rec_mean"] * 100) / 100);
    $("#quad_f1").text(Math.round(data["res_quad"]["f1_mean"] * 100) / 100);
    $("#quad_acc").text(Math.round(data["res_quad"]["acc_mean"] * 100) / 100);
}

function executeProcess(data) {
    fetch('process_dataset', {
        method: 'POST',
        body: JSON.stringify(data),
        headers:{ 'Content-Type': 'application/json' }
    }).then(res => res.json())
    .catch(error => console.error('Error:', error))
    .then(function(response) {
        console.log("Executing", response);
        circumplexMan.plotPoints(response['class'], response['emo_names']);
        $("#conf_matrix canvas").remove();
        circumplexMan.plotConfusionMatrix("#conf_matrix", response['emo_names'], response["class_gt"], response ["class"]);
        featureManager.plotFeatures(response['features']);

        // Show list of features
        sampleSize.value = response["sampleSize"];
        $("#listFeatures option").remove();
        var feat_names = response["features"]["feat_names"];
        feat_names.sort();
        for (var i in feat_names) {
            $("#listFeatures").append("<option value='" + feat_names[i] + "'>" + feat_names[i] + "</option>");
        }
        showMetrics(response["metrics"]);
    });
}

function executeRetrain(data) {
    fetch('retrain_classifier', {
        method: 'POST',
        body: JSON.stringify(data),
        headers:{ 'Content-Type': 'application/json' }
    }).then(res => res.json())
    .catch(error => console.error('Error:', error))
    .then(function(response) {
        /*console.log("Executing", response);
        circumplexMan.plotPoints(response['class'], response['emo_names']);
        featureManager.plotFeatures(response['features']);

        // Show list of features
        sampleSize.value = response["sampleSize"];
        $("#listFeatures option").remove();
        var feat_names = response["features"]["feat_names"];
        feat_names.sort();
        for (var i in feat_names) {
            $("#listFeatures").append("<option value='" + feat_names[i] + "'>" + feat_names[i] + "</option>");
        }*/
        showMetrics(response["metrics"]);
    });
}