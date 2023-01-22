function readTextFile(file, callback) {
    var rawFile = new XMLHttpRequest();
    rawFile.overrideMimeType("application/json");
    rawFile.open("GET", file, true);
    rawFile.onreadystatechange = function() {
        if (rawFile.readyState === 4 && rawFile.status == "200") {
            callback(rawFile.responseText);
        }
    }
    rawFile.send(null);
}


chrome.runtime.onInstalled.addListener(function() {
  fetch('../gt_data.json')
    .then(
      function(response) {
        if (response.status !== 200) {
          console.log("Data loading problem: " + response.status);
          return;
        }

        response.json().then(function(data) {
          console.log(data);
          chrome.storage.local.set({'alternative_data': data}, function() {
            console.log('data loaded');
          });
        });
      }
    )
    .catch(function(err) {
      console.log("Fetch Error: -S", err);
    });
});




function loadAlternativeInformation() {

  chrome.runtime.onInstalled.addListener(function() {
    readTextFile("gt_data.json", function(text){
      var data = JSON.parse(text);
      chrome.storage.local.set({'alternative_data': data}, function() {
        console.log("loaded!");
      });
    });
  });

}
var bg = this;

chrome.runtime.onMessage.addListener(
  function(request, sender, sendResponse) {
    if (request.action === 'exec' && bg[request.payload.fn]) {
      bg[request.payload.fn].call(bg, ...request.payload.args);
    }
  });
