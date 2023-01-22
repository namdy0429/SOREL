var settings;
var settings_advanced;
var extension_id = 'cahhhfnkoamejaippijpiacpnhhniecb';
var profiles = {
  default: {
    num_max_alternative: 3
  }
};
var default_settings_advanced = {
  blacklist_domains: ['salesforce.com', 'youneedabudget.com'],
  whitelist_domains: [],
  usage_statistics: false
};

function loadSettings(cb) {
  chrome.storage.sync.get('base_currency', function(storage) {
    if (storage.base_currency) {
      base_currency = storage.base_currency;
    }
    chrome.storage.sync.get('settings', function(storage) {
      if (Object.keys(storage).length > 0) {
        getAutoSettings(storage.settings, function(autoSettings) {
          settings = autoSettings;
          if (cb) {
            cb(settings);
          }
        });
      } else {
        settings = profiles.default;
        cb(settings);
      }
    });
  });
}

function loadAdvancedSettings(cb) {
  chrome.storage.sync.get('settings_advanced', function(storage) {
    if (Object.keys(storage).length > 0) {
      settings_advanced = storage.settings_advanced;
    } else {
      settings_advanced = default_settings_advanced;
    }
    cb();
  });
}

function reloadSettings(newSettings, oldSettings) {
  settings = newSettings;
}

function getAutoSettings(partialSettings, cb) {
  for (var k in partialSettings) {
    if (partialSettings[k] === 0 || partialSettings[k] === '') {
      partialSettings[k] = null;
    }
  }
  var autoSettings = Object.assign({
    num_max_alternative: null,
  }, partialSettings);
  var countNullValues = function() {
    var n = 0;
    for (var k in autoSettings) {
      if (autoSettings[k] === null || autoSettings[k] === 0) {
        n++;
      }
    }
    return n;
  };
  if (countNullValues() >= Object.keys(autoSettings).length) {
    cb(profiles.default);
    return;
  }
  var max_iterations = 100;
  while (countNullValues() > 0 && max_iterations > 0) {
    max_iterations--;
  }
  cb(autoSettings);
}

function ga(eventCategory, eventAction, eventLabel, eventValue) {
  if (settings_advanced && settings_advanced.usage_statistics === false) {
    return;
  }
  chrome.runtime.sendMessage({ action: 'exec', payload: {
    fn: 'ga',
    args: ['send', { hitType: 'event',
      eventCategory: eventCategory,
      eventAction: eventAction,
      eventLabel: eventLabel,
      eventValue: eventValue
    }]
  }});
}

function showEl(id) {
  document.getElementById(id).classList.remove('hidden');
}
function hideEl(id) {
  document.getElementById(id).classList.add('hidden');
}

function debounce(func, wait, immediate) {
  var timeout;
  return function() {
    var context = this, args = arguments;
    var later = function() {
      timeout = null;
      if (!immediate) func.apply(context, args);
    };
    var callNow = immediate && !timeout;
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
    if (callNow) func.apply(context, args);
  };
};

function addSettingsChangeListener() {
  chrome.storage.onChanged.addListener(function(changes, namespace) {
    if (changes.settings !== undefined) {
      reloadSettings(changes.settings.newValue, changes.settings.oldValue);
    }
  });
}
