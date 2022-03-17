(function () {
    'use strict'

    var winHeight = window.innerHeight/2 - 55;
    $("#circumplex").height(winHeight + "px");
    $("#heatmap").height(winHeight + "px");

    $('#dataset').change(function() {
        $("#channels option").remove();
        if ($(this).val() != "") {
            var data = {'dataset': $(this).val()};
            loadSignals(data, $("#channels"));
        }
    });

    $('#listFeatures').change(function() {
        if ($(this).val().length > 0)
            $("#btnRetrain").removeAttr("disabled");
        else
            $("#btnRetrain").attr("disabled", "disabled");
    });

    $("#btnExec").click(function() {
        var tmp_mode = $("#mode").val().split("_")
        var data = {
            'dataset': $("#dataset").val(),
            'fselector': $("#fselector").val(),
            'classifier': $("#classifier").val(),
            'mode': tmp_mode[0],
            'nClasses': Number(tmp_mode[1]),
            'winSize': Number($("#winSize").val()),
            'winIni': Number($("#winIni").val()),
            'sampleSize': Number($("#sampleSize").val()),
            'channels': $("#channels").val(),
            'testSize': 20 //Number($("#testSize").val())
        };
        if(data['dataset'] != "" && data['classifier'] != "" && !isNaN(data["winSize"]) && data["channels"].length > 0) {
            executeProcess(data);
        }
        else
            console.log("Some parameters is needed.")
    });

    $("#btnRetrain").click(function() {
        var tmp_mode = $("#mode").val().split("_")
        var data = {
            'dataset': $("#dataset").val(),
            'classifier': $("#classifier").val(),
            'fselector': "",
            'nClasses': Number(tmp_mode[1]),
            'winSize': Number($("#winSize").val()),
            'winIni': Number($("#winIni").val()),
            'sampleSize': Number($("#sampleSize").val()),
            'features': $("#listFeatures").val()
        };
        if(data['dataset'] != "" && data['classifier'] != "" && data["features"].length > 0) {
            executeRetrain(data);
        }
        else
            console.log("Some parameters are needed.")
    });
}())
