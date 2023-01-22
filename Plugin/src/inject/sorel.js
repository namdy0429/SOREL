var namespace = 'sorel-';
var re_result;

function getAllElements(addedNodes) {
  var elements = [];
  for (var i = 0; i < addedNodes.length; i++) {
    elements.push(addedNodes[i]);
    if (addedNodes[i].nodeType === 1) {
      var children = addedNodes[i].querySelectorAll(':not(textarea):not([type=text])');
      for (var j = 0; j < children.length; j++) {
        elements.push(children[j]);
      }
    }
  }
  var results = [];
  var child;
  for(var i = 0; i < elements.length; i++) {
    if (elements[i].childNodes) {
      for(var j = 0; j < elements[i].childNodes.length; j++) {
        child = elements[i].childNodes[j];
        if((elements[i].hasChildNodes() && child.nodeType === 3)) {
          results.push(child);
        }
      }
    }
  }
  return results;
}

var tooltip = null;
var hovering_tooltip = false;

function createTooltip() {
  tooltip = document.getElementById(namespace + 'tooltip')
  if (tooltip !== null) {
    return;
  }
  tooltip = document.createElement('div');
  tooltip.id = namespace + 'tooltip';
  tooltip.innerHTML = `<span class="${namespace}content"></span>`;
  document.body.append(tooltip);
}

function showTooltip(ev) {
  var el = ev.target;
  // Ugly way to look for up to 2 parents (for Amazon)
  if (!el.hasAttribute(namespace + 'text')) {
    if (el.parentNode && el.parentNode.hasAttribute(namespace + 'text')) {
      el = el.parentNode;
    } else {
      if (el.parentNode && el.parentNode.parentNode && el.parentNode.parentNode.hasAttribute(namespace + 'text')) {
        el = el.parentNode.parentNode;
      } else {
        return;
      }
    }
  }
  var content_el = tooltip.querySelector('.' + namespace + 'content');
  content_el.innerHTML = el.getAttribute(namespace + 'text');
  var cords = el.getBoundingClientRect();
  var content_cords = content_el.getBoundingClientRect();
  var top, left, ttClass;
  var padding = 20;
  if (cords.left < content_cords.width + 5) {
    // Right
    top = Math.round(cords.top + cords.height/2 - content_cords.height/2) + 'px';
    left = (Math.round(cords.right) + padding) + 'px'
    ttClass = namespace + 'right';
  } else if (cords.right > window.innerWidth - content_cords.width/2 - 5) {
    // Left
    top = Math.round(cords.top + cords.height/2 - content_cords.height/2) + 'px';
    left = Math.round(cords.left - content_cords.width - padding) + 'px'
    ttClass = namespace + 'left';
  } else if (cords.top < content_cords.height + padding + 5) {
    // Bottom
    top = (cords.bottom + padding) + 'px';
    left = Math.round(cords.left + cords.width/2 - content_cords.width/2) + 'px';
    ttClass = namespace + 'bottom';
  } else {
    // Top
    top = (cords.top - content_cords.height - padding) + 'px';
    left = Math.round(cords.left + cords.width/2 - content_cords.width/2) + 'px';
    ttClass = namespace + 'top';
  }
  tooltip.style.left = left;
  tooltip.style.top = top;
  tooltip.classList.remove(namespace + 'right');
  tooltip.classList.remove(namespace + 'left');
  tooltip.classList.remove(namespace + 'top');
  tooltip.classList.remove(namespace + 'bottom');
  tooltip.classList.add(ttClass);
  tooltip.classList.add('visible');
  tooltip.addEventListener('mouseenter', function() { hovering_tooltip = true;}, false);
  tooltip.addEventListener('mouseleave', function(){ hideTooltip(); hovering_tooltip = false;}, false);
}
function hideTooltip() {
  tooltip.classList.remove('visible');
}

var re = new RegExp('tf\.([^}]+?)(?=[^a-zA-Z0-9.,_}]|$)');
var re_short = new RegExp('tf');
var re_function = function(original, sign, amount) {
  if (original in re_result) {
    var keys = Object.keys(re_result[original]);
    var content = ""
    for (var key_i=0; key_i<Object.keys(re_result[original]).length;key_i++) {
      // for (var key_i=0; key_i<Math.min(Object.keys(re_result[original]).length, 3);key_i++) {
    // for (var key_i=0; key_i<1;key_i++) {
      let i = Object.keys(re_result[original])[key_i];
      let h = re_result[original][i][0]["h"].trim();
      let t = re_result[original][i][0]["t"].trim();
      let title = re_result[original][i][0]["title"].trim();
      let alter = "";
      if (original == h) {
        alter = t;
      }
      else {
        alter = h;
      }
      let ori_link = original.replaceAll(`.`, `/`);
      let alter_link = alter.replaceAll(`.`, `/`);
      let s = re_result[original][i][0]["sentences"];
      let ss = s.join(` `);
      if (h.length >= t.length) {
        var s_re = new RegExp(h, 'g');
        ss = ss.replace(s_re, `<b>`+h+`</b>`);
        s_re = new RegExp(t, 'g');
        ss = ss.replace(s_re, `<b>`+t+`</b>`)
      }
      else {
        var s_re = new RegExp(t, 'g');
        ss = ss.replace(s_re, `<b>`+t+`</b>`);
        s_re = new RegExp(h, 'g');
        ss = ss.replace(s_re, `<b>`+h+`</b>`)
      }


      content += `<hr>`;
      content += `<b><a href='https://www.tensorflow.org/api_docs/python/` + alter_link + `' target='_blank' rel='noopener noreferrer'>` + alter + `</a></br>`;
      content += ` is an alternative of `;
      content += `<br><a href='https://www.tensorflow.org/api_docs/python/` + ori_link + `' target='_blank' rel='noopener noreferrer'>` + original + `</a></b>`;
      // content += `<hr>&#8618;`;
      content += `<hr>`;
      // content += s.join(` `);
      content += ss;
      content += `<br><a href='https://www.stackoverflow.com/questions/` + title + `' target='_blank' rel='noopener noreferrer'>See original Stack Overflow post</a>`;
      content += `<hr>`;
      content = content.replaceAll(`"`, `'`);
    }

    return {
      original: original,
      content: content,
      sentence: ""
    };
  }
  else {
    return {
      original: original,
      content: ""
    }
  }
};
var html_tag = 'span-tp';
function findMethods(addedNodes) {
  createTooltip();
  var nodes = getAllElements(addedNodes);
  for (var i = 0, len = nodes.length; i < len; i++) {
    var el = nodes[i];

    if (!el || !el.parentNode || el.parentNode.hasAttribute(namespace + 'text')) {
      continue;
    }

    if (el.nodeValue && el.nodeValue.match(re)) {
      var new_html = el.parentNode.textContent.replace(re, function(original, sign, amount) {
        var matches = re_function(original, sign, amount);
        if (matches.content === "") {
          return `${matches.original}`;
        }
        else {
          return `<${html_tag} ${namespace}text="${matches.content}" ${namespace}isnew="true" >${matches.original}<sup><small>&#127386;</small></sup></${html_tag}>`;
        }
      });
      el.parentNode.innerHTML = new_html;
    }
  }
  removeDuplicates();
  attachHoverEvents();
}

function removeDuplicates() {
  // Leave only the most deep instances
  var elements = document.querySelectorAll('[' + namespace + 'text]');
  for(var i = 0; i < elements.length; i++) {
    var el = elements[i];
    var parent = el.parentNode;
    var max_depth = 50;
    while (max_depth > 0 && parent && parent !== document) {
      if (parent.hasAttribute(namespace + 'text')) {
        parent.removeAttribute(namespace + 'text');
        parent.removeAttribute(namespace + 'isnew');
      }
      parent = parent.parentNode;
      max_depth--;
    }
  }
}

function attachHoverEvents() {
  var elements = document.querySelectorAll('[' + namespace + 'isnew="true"]');
  for(var i = 0; i < elements.length; i++) {
    var el = elements[i];
    el.addEventListener('mouseenter', function(ev) {
      showTooltip(ev);
    });
    el.addEventListener('mouseleave', function(ev){
      setTimeout(function() {
        if (!hovering_tooltip) {
          hideTooltip();
          hovering_tooltip = false;
        }
      }, 500);
    }, false);
    el.setAttribute(namespace + 'isnew', 'false');
  }
}

var addedNodes = [];
// Rapid DOM changes may be missed because of debounce. It's a tradeoff for not overloading the user's CPU on heavy DOM manipulating websites
var handleDomChanges = debounce(function() {
  findMethods(addedNodes);
  addedNodes = [];
}, 150);

function listenForDomChanges() {
  var target = document.querySelector('body div:not([id^='+namespace+'])');
  var observer = new MutationObserver(function(mutations) {
    for (var i = 0; i < mutations.length; i++) {
      var nodes = mutations[i].addedNodes;
      if (nodes.length > 0) {
        for (var j = 0; j < nodes.length; j++) {
          addedNodes.push(nodes[j]);
        }
      }
    }
    handleDomChanges();
  });
  var config = { childList: true, subtree: true };
  observer.observe(target, config);
}

// For the rare case that the DOM is done loading before the extension
function triggerDomChange() {
  addedNodes.push(document.body);
  handleDomChanges();
}

function startTimeprices(cb) {
  loadAdvancedSettings(function() {
    var current_domain = window.location.hostname;
    if (settings_advanced.whitelist_domains.length > 0) {
      for (var i = 0; i < settings_advanced.whitelist_domains.length; i++) {
        var domain_regex = new RegExp(settings_advanced.whitelist_domains[i].replace('.', '\.') + '$', 'i');
        if (current_domain.match(domain_regex)) {
          cb();
          return;
        }
      }
    } else {
      for (var i = 0; i < settings_advanced.blacklist_domains.length; i++) {
        var domain_regex = new RegExp(settings_advanced.blacklist_domains[i].replace('.', '\.') + '$', 'i');
        if (current_domain.match(domain_regex)) {
          return;
        }
      }
      cb();
    }
  });
}

startTimeprices(function() {
  loadSettings(function() {
    chrome.storage.local.get('alternative_data', function(result){
      re_result = result.alternative_data;
      listenForDomChanges();
      triggerDomChange();
      addSettingsChangeListener();
    });
  });
});
