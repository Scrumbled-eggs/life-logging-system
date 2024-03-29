// modules are defined as an array
// [ module function, map of requires ]
//
// map of requires is short require name -> numeric require
//
// anything defined in a previous bundle is accessed via the
// orig method which is the require for previous bundles

(function (modules, entry, mainEntry, parcelRequireName, globalName) {
  /* eslint-disable no-undef */
  var globalObject =
    typeof globalThis !== 'undefined'
      ? globalThis
      : typeof self !== 'undefined'
      ? self
      : typeof window !== 'undefined'
      ? window
      : typeof global !== 'undefined'
      ? global
      : {};
  /* eslint-enable no-undef */

  // Save the require from previous bundle to this closure if any
  var previousRequire =
    typeof globalObject[parcelRequireName] === 'function' &&
    globalObject[parcelRequireName];

  var cache = previousRequire.cache || {};
  // Do not use `require` to prevent Webpack from trying to bundle this call
  var nodeRequire =
    typeof module !== 'undefined' &&
    typeof module.require === 'function' &&
    module.require.bind(module);

  function newRequire(name, jumped) {
    if (!cache[name]) {
      if (!modules[name]) {
        // if we cannot find the module within our internal map or
        // cache jump to the current global require ie. the last bundle
        // that was added to the page.
        var currentRequire =
          typeof globalObject[parcelRequireName] === 'function' &&
          globalObject[parcelRequireName];
        if (!jumped && currentRequire) {
          return currentRequire(name, true);
        }

        // If there are other bundles on this page the require from the
        // previous one is saved to 'previousRequire'. Repeat this as
        // many times as there are bundles until the module is found or
        // we exhaust the require chain.
        if (previousRequire) {
          return previousRequire(name, true);
        }

        // Try the node require function if it exists.
        if (nodeRequire && typeof name === 'string') {
          return nodeRequire(name);
        }

        var err = new Error("Cannot find module '" + name + "'");
        err.code = 'MODULE_NOT_FOUND';
        throw err;
      }

      localRequire.resolve = resolve;
      localRequire.cache = {};

      var module = (cache[name] = new newRequire.Module(name));

      modules[name][0].call(
        module.exports,
        localRequire,
        module,
        module.exports,
        this
      );
    }

    return cache[name].exports;

    function localRequire(x) {
      var res = localRequire.resolve(x);
      return res === false ? {} : newRequire(res);
    }

    function resolve(x) {
      var id = modules[name][1][x];
      return id != null ? id : x;
    }
  }

  function Module(moduleName) {
    this.id = moduleName;
    this.bundle = newRequire;
    this.exports = {};
  }

  newRequire.isParcelRequire = true;
  newRequire.Module = Module;
  newRequire.modules = modules;
  newRequire.cache = cache;
  newRequire.parent = previousRequire;
  newRequire.register = function (id, exports) {
    modules[id] = [
      function (require, module) {
        module.exports = exports;
      },
      {},
    ];
  };

  Object.defineProperty(newRequire, 'root', {
    get: function () {
      return globalObject[parcelRequireName];
    },
  });

  globalObject[parcelRequireName] = newRequire;

  for (var i = 0; i < entry.length; i++) {
    newRequire(entry[i]);
  }

  if (mainEntry) {
    // Expose entry point to Node, AMD or browser globals
    // Based on https://github.com/ForbesLindesay/umd/blob/master/template.js
    var mainExports = newRequire(mainEntry);

    // CommonJS
    if (typeof exports === 'object' && typeof module !== 'undefined') {
      module.exports = mainExports;

      // RequireJS
    } else if (typeof define === 'function' && define.amd) {
      define(function () {
        return mainExports;
      });

      // <script>
    } else if (globalName) {
      this[globalName] = mainExports;
    }
  }
})({"jKwHT":[function(require,module,exports) {
"use strict";
var HMR_HOST = null;
var HMR_PORT = null;
var HMR_SECURE = false;
var HMR_ENV_HASH = "d6ea1d42532a7575";
module.bundle.HMR_BUNDLE_ID = "fe4256060641b553";
function _toConsumableArray(arr) {
    return _arrayWithoutHoles(arr) || _iterableToArray(arr) || _unsupportedIterableToArray(arr) || _nonIterableSpread();
}
function _nonIterableSpread() {
    throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
}
function _iterableToArray(iter) {
    if (typeof Symbol !== "undefined" && iter[Symbol.iterator] != null || iter["@@iterator"] != null) return Array.from(iter);
}
function _arrayWithoutHoles(arr) {
    if (Array.isArray(arr)) return _arrayLikeToArray(arr);
}
function _createForOfIteratorHelper(o, allowArrayLike) {
    var it = typeof Symbol !== "undefined" && o[Symbol.iterator] || o["@@iterator"];
    if (!it) {
        if (Array.isArray(o) || (it = _unsupportedIterableToArray(o)) || allowArrayLike && o && typeof o.length === "number") {
            if (it) o = it;
            var i = 0;
            var F = function F() {};
            return {
                s: F,
                n: function n() {
                    if (i >= o.length) return {
                        done: true
                    };
                    return {
                        done: false,
                        value: o[i++]
                    };
                },
                e: function e(_e) {
                    throw _e;
                },
                f: F
            };
        }
        throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
    }
    var normalCompletion = true, didErr = false, err;
    return {
        s: function s() {
            it = it.call(o);
        },
        n: function n() {
            var step = it.next();
            normalCompletion = step.done;
            return step;
        },
        e: function e(_e2) {
            didErr = true;
            err = _e2;
        },
        f: function f() {
            try {
                if (!normalCompletion && it.return != null) it.return();
            } finally{
                if (didErr) throw err;
            }
        }
    };
}
function _unsupportedIterableToArray(o, minLen) {
    if (!o) return;
    if (typeof o === "string") return _arrayLikeToArray(o, minLen);
    var n = Object.prototype.toString.call(o).slice(8, -1);
    if (n === "Object" && o.constructor) n = o.constructor.name;
    if (n === "Map" || n === "Set") return Array.from(o);
    if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen);
}
function _arrayLikeToArray(arr, len) {
    if (len == null || len > arr.length) len = arr.length;
    for(var i = 0, arr2 = new Array(len); i < len; i++)arr2[i] = arr[i];
    return arr2;
}
/* global HMR_HOST, HMR_PORT, HMR_ENV_HASH, HMR_SECURE, chrome, browser */ /*::
import type {
  HMRAsset,
  HMRMessage,
} from '@parcel/reporter-dev-server/src/HMRServer.js';
interface ParcelRequire {
  (string): mixed;
  cache: {|[string]: ParcelModule|};
  hotData: mixed;
  Module: any;
  parent: ?ParcelRequire;
  isParcelRequire: true;
  modules: {|[string]: [Function, {|[string]: string|}]|};
  HMR_BUNDLE_ID: string;
  root: ParcelRequire;
}
interface ParcelModule {
  hot: {|
    data: mixed,
    accept(cb: (Function) => void): void,
    dispose(cb: (mixed) => void): void,
    // accept(deps: Array<string> | string, cb: (Function) => void): void,
    // decline(): void,
    _acceptCallbacks: Array<(Function) => void>,
    _disposeCallbacks: Array<(mixed) => void>,
  |};
}
interface ExtensionContext {
  runtime: {|
    reload(): void,
  |};
}
declare var module: {bundle: ParcelRequire, ...};
declare var HMR_HOST: string;
declare var HMR_PORT: string;
declare var HMR_ENV_HASH: string;
declare var HMR_SECURE: boolean;
declare var chrome: ExtensionContext;
declare var browser: ExtensionContext;
*/ var OVERLAY_ID = '__parcel__error__overlay__';
var OldModule = module.bundle.Module;
function Module(moduleName) {
    OldModule.call(this, moduleName);
    this.hot = {
        data: module.bundle.hotData,
        _acceptCallbacks: [],
        _disposeCallbacks: [],
        accept: function accept(fn) {
            this._acceptCallbacks.push(fn || function() {});
        },
        dispose: function dispose(fn) {
            this._disposeCallbacks.push(fn);
        }
    };
    module.bundle.hotData = undefined;
}
module.bundle.Module = Module;
var checkedAssets, acceptedAssets, assetsToAccept /*: Array<[ParcelRequire, string]> */ ;
function getHostname() {
    return HMR_HOST || (location.protocol.indexOf('http') === 0 ? location.hostname : 'localhost');
}
function getPort() {
    return HMR_PORT || location.port;
} // eslint-disable-next-line no-redeclare
var parent = module.bundle.parent;
if ((!parent || !parent.isParcelRequire) && typeof WebSocket !== 'undefined') {
    var hostname = getHostname();
    var port = getPort();
    var protocol = HMR_SECURE || location.protocol == 'https:' && !/localhost|127.0.0.1|0.0.0.0/.test(hostname) ? 'wss' : 'ws';
    var ws = new WebSocket(protocol + '://' + hostname + (port ? ':' + port : '') + '/'); // $FlowFixMe
    ws.onmessage = function(event) {
        checkedAssets = {} /*: {|[string]: boolean|} */ ;
        acceptedAssets = {} /*: {|[string]: boolean|} */ ;
        assetsToAccept = [];
        var data = JSON.parse(event.data);
        if (data.type === 'update') {
            // Remove error overlay if there is one
            if (typeof document !== 'undefined') removeErrorOverlay();
            var assets = data.assets.filter(function(asset) {
                return asset.envHash === HMR_ENV_HASH;
            }); // Handle HMR Update
            var handled = assets.every(function(asset) {
                return asset.type === 'css' || asset.type === 'js' && hmrAcceptCheck(module.bundle.root, asset.id, asset.depsByBundle);
            });
            if (handled) {
                console.clear();
                assets.forEach(function(asset) {
                    hmrApply(module.bundle.root, asset);
                });
                for(var i = 0; i < assetsToAccept.length; i++){
                    var id = assetsToAccept[i][1];
                    if (!acceptedAssets[id]) hmrAcceptRun(assetsToAccept[i][0], id);
                }
            } else if ('reload' in location) location.reload();
            else {
                // Web extension context
                var ext = typeof chrome === 'undefined' ? typeof browser === 'undefined' ? null : browser : chrome;
                if (ext && ext.runtime && ext.runtime.reload) ext.runtime.reload();
            }
        }
        if (data.type === 'error') {
            // Log parcel errors to console
            var _iterator = _createForOfIteratorHelper(data.diagnostics.ansi), _step;
            try {
                for(_iterator.s(); !(_step = _iterator.n()).done;){
                    var ansiDiagnostic = _step.value;
                    var stack = ansiDiagnostic.codeframe ? ansiDiagnostic.codeframe : ansiDiagnostic.stack;
                    console.error('🚨 [parcel]: ' + ansiDiagnostic.message + '\n' + stack + '\n\n' + ansiDiagnostic.hints.join('\n'));
                }
            } catch (err) {
                _iterator.e(err);
            } finally{
                _iterator.f();
            }
            if (typeof document !== 'undefined') {
                // Render the fancy html overlay
                removeErrorOverlay();
                var overlay = createErrorOverlay(data.diagnostics.html); // $FlowFixMe
                document.body.appendChild(overlay);
            }
        }
    };
    ws.onerror = function(e) {
        console.error(e.message);
    };
    ws.onclose = function() {
        console.warn('[parcel] 🚨 Connection to the HMR server was lost');
    };
}
function removeErrorOverlay() {
    var overlay = document.getElementById(OVERLAY_ID);
    if (overlay) {
        overlay.remove();
        console.log('[parcel] ✨ Error resolved');
    }
}
function createErrorOverlay(diagnostics) {
    var overlay = document.createElement('div');
    overlay.id = OVERLAY_ID;
    var errorHTML = '<div style="background: black; opacity: 0.85; font-size: 16px; color: white; position: fixed; height: 100%; width: 100%; top: 0px; left: 0px; padding: 30px; font-family: Menlo, Consolas, monospace; z-index: 9999;">';
    var _iterator2 = _createForOfIteratorHelper(diagnostics), _step2;
    try {
        for(_iterator2.s(); !(_step2 = _iterator2.n()).done;){
            var diagnostic = _step2.value;
            var stack = diagnostic.codeframe ? diagnostic.codeframe : diagnostic.stack;
            errorHTML += "\n      <div>\n        <div style=\"font-size: 18px; font-weight: bold; margin-top: 20px;\">\n          \uD83D\uDEA8 ".concat(diagnostic.message, "\n        </div>\n        <pre>").concat(stack, "</pre>\n        <div>\n          ").concat(diagnostic.hints.map(function(hint) {
                return '<div>💡 ' + hint + '</div>';
            }).join(''), "\n        </div>\n        ").concat(diagnostic.documentation ? "<div>\uD83D\uDCDD <a style=\"color: violet\" href=\"".concat(diagnostic.documentation, "\" target=\"_blank\">Learn more</a></div>") : '', "\n      </div>\n    ");
        }
    } catch (err) {
        _iterator2.e(err);
    } finally{
        _iterator2.f();
    }
    errorHTML += '</div>';
    overlay.innerHTML = errorHTML;
    return overlay;
}
function getParents(bundle, id) /*: Array<[ParcelRequire, string]> */ {
    var modules = bundle.modules;
    if (!modules) return [];
    var parents = [];
    var k, d, dep;
    for(k in modules)for(d in modules[k][1]){
        dep = modules[k][1][d];
        if (dep === id || Array.isArray(dep) && dep[dep.length - 1] === id) parents.push([
            bundle,
            k
        ]);
    }
    if (bundle.parent) parents = parents.concat(getParents(bundle.parent, id));
    return parents;
}
function updateLink(link) {
    var newLink = link.cloneNode();
    newLink.onload = function() {
        if (link.parentNode !== null) // $FlowFixMe
        link.parentNode.removeChild(link);
    };
    newLink.setAttribute('href', link.getAttribute('href').split('?')[0] + '?' + Date.now()); // $FlowFixMe
    link.parentNode.insertBefore(newLink, link.nextSibling);
}
var cssTimeout = null;
function reloadCSS() {
    if (cssTimeout) return;
    cssTimeout = setTimeout(function() {
        var links = document.querySelectorAll('link[rel="stylesheet"]');
        for(var i = 0; i < links.length; i++){
            // $FlowFixMe[incompatible-type]
            var href = links[i].getAttribute('href');
            var hostname = getHostname();
            var servedFromHMRServer = hostname === 'localhost' ? new RegExp('^(https?:\\/\\/(0.0.0.0|127.0.0.1)|localhost):' + getPort()).test(href) : href.indexOf(hostname + ':' + getPort());
            var absolute = /^https?:\/\//i.test(href) && href.indexOf(location.origin) !== 0 && !servedFromHMRServer;
            if (!absolute) updateLink(links[i]);
        }
        cssTimeout = null;
    }, 50);
}
function hmrApply(bundle, asset) {
    var modules = bundle.modules;
    if (!modules) return;
    if (asset.type === 'css') reloadCSS();
    else if (asset.type === 'js') {
        var deps = asset.depsByBundle[bundle.HMR_BUNDLE_ID];
        if (deps) {
            if (modules[asset.id]) {
                // Remove dependencies that are removed and will become orphaned.
                // This is necessary so that if the asset is added back again, the cache is gone, and we prevent a full page reload.
                var oldDeps = modules[asset.id][1];
                for(var dep in oldDeps)if (!deps[dep] || deps[dep] !== oldDeps[dep]) {
                    var id = oldDeps[dep];
                    var parents = getParents(module.bundle.root, id);
                    if (parents.length === 1) hmrDelete(module.bundle.root, id);
                }
            }
            var fn = new Function('require', 'module', 'exports', asset.output);
            modules[asset.id] = [
                fn,
                deps
            ];
        } else if (bundle.parent) hmrApply(bundle.parent, asset);
    }
}
function hmrDelete(bundle, id1) {
    var modules = bundle.modules;
    if (!modules) return;
    if (modules[id1]) {
        // Collect dependencies that will become orphaned when this module is deleted.
        var deps = modules[id1][1];
        var orphans = [];
        for(var dep in deps){
            var parents = getParents(module.bundle.root, deps[dep]);
            if (parents.length === 1) orphans.push(deps[dep]);
        } // Delete the module. This must be done before deleting dependencies in case of circular dependencies.
        delete modules[id1];
        delete bundle.cache[id1]; // Now delete the orphans.
        orphans.forEach(function(id) {
            hmrDelete(module.bundle.root, id);
        });
    } else if (bundle.parent) hmrDelete(bundle.parent, id1);
}
function hmrAcceptCheck(bundle, id, depsByBundle) {
    if (hmrAcceptCheckOne(bundle, id, depsByBundle)) return true;
     // Traverse parents breadth first. All possible ancestries must accept the HMR update, or we'll reload.
    var parents = getParents(module.bundle.root, id);
    var accepted = false;
    while(parents.length > 0){
        var v = parents.shift();
        var a = hmrAcceptCheckOne(v[0], v[1], null);
        if (a) // If this parent accepts, stop traversing upward, but still consider siblings.
        accepted = true;
        else {
            // Otherwise, queue the parents in the next level upward.
            var p = getParents(module.bundle.root, v[1]);
            if (p.length === 0) {
                // If there are no parents, then we've reached an entry without accepting. Reload.
                accepted = false;
                break;
            }
            parents.push.apply(parents, _toConsumableArray(p));
        }
    }
    return accepted;
}
function hmrAcceptCheckOne(bundle, id, depsByBundle) {
    var modules = bundle.modules;
    if (!modules) return;
    if (depsByBundle && !depsByBundle[bundle.HMR_BUNDLE_ID]) {
        // If we reached the root bundle without finding where the asset should go,
        // there's nothing to do. Mark as "accepted" so we don't reload the page.
        if (!bundle.parent) return true;
        return hmrAcceptCheck(bundle.parent, id, depsByBundle);
    }
    if (checkedAssets[id]) return true;
    checkedAssets[id] = true;
    var cached = bundle.cache[id];
    assetsToAccept.push([
        bundle,
        id
    ]);
    if (!cached || cached.hot && cached.hot._acceptCallbacks.length) return true;
}
function hmrAcceptRun(bundle, id) {
    var cached = bundle.cache[id];
    bundle.hotData = {};
    if (cached && cached.hot) cached.hot.data = bundle.hotData;
    if (cached && cached.hot && cached.hot._disposeCallbacks.length) cached.hot._disposeCallbacks.forEach(function(cb) {
        cb(bundle.hotData);
    });
    delete bundle.cache[id];
    bundle(id);
    cached = bundle.cache[id];
    if (cached && cached.hot && cached.hot._acceptCallbacks.length) cached.hot._acceptCallbacks.forEach(function(cb) {
        var assetsToAlsoAccept = cb(function() {
            return getParents(module.bundle.root, id);
        });
        if (assetsToAlsoAccept && assetsToAccept.length) // $FlowFixMe[method-unbinding]
        assetsToAccept.push.apply(assetsToAccept, assetsToAlsoAccept);
    });
    acceptedAssets[id] = true;
}

},{}],"bNKaB":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
var _instantsearchJs = require("instantsearch.js");
var _instantsearchJsDefault = parcelHelpers.interopDefault(_instantsearchJs);
var _lite = require("algoliasearch/lite");
var _liteDefault = parcelHelpers.interopDefault(_lite);
var _autocompleteJs = require("@algolia/autocomplete-js");
var _connectors = require("instantsearch.js/es/connectors");
var _history = require("instantsearch.js/es/lib/routers/history");
var _historyDefault = parcelHelpers.interopDefault(_history);
var _autocompletePluginQuerySuggestions = require("@algolia/autocomplete-plugin-query-suggestions");
var _autocompletePluginRecentSearches = require("@algolia/autocomplete-plugin-recent-searches");
var _preact = require("preact");
var _widgets = require("instantsearch.js/es/widgets");
var _autocompleteThemeClassic = require("@algolia/autocomplete-theme-classic");
const searchClient = _liteDefault.default('0L0TPDZHFM', '1a42927a7a1ffc3661c466e3a7acda87');
const INSTANT_SEARCH_INDEX_NAME = 'milestone3';
const instantSearchRouter = _historyDefault.default();
const search = _instantsearchJsDefault.default({
    indexName: INSTANT_SEARCH_INDEX_NAME,
    searchClient,
    routing: instantSearchRouter
});
const virtualSearchBox = _connectors.connectSearchBox(()=>{});
search.addWidgets([
    _widgets.refinementList({
        container: '#face-attr',
        attribute: 'face_attributes',
        showMore: true
    }),
    _widgets.rangeSlider({
        container: '#filter-list',
        attribute: 'filelength(s)'
    }),
    _widgets.rangeSlider({
        container: '#fps-list',
        attribute: 'fps'
    }),
    _widgets.configure({
        hitsPerPage: 16
    }),
    virtualSearchBox({}),
    _widgets.hits({
        container: '#hits',
        templates: {
            item: `
<div>
      <div class="hit-face">
        {{#helpers.highlight}}{ "attribute": "face" }{{/helpers.highlight}}
      </div>
      <div class="hit-action">
        {{#helpers.highlight}}{ "attribute": "action" }{{/helpers.highlight}}
      </div>
      <div class="hit-file">
      filename:
      {{#helpers.highlight}}{ "attribute": "filename" }{{/helpers.highlight}}
      </div>
      <div class="hit-frame">frame:{{frame}}</div>
      <div class="hit-timestamp">timestamp(s):{{timestamp(s)}}</div>
      <div class="hit-filelen">filelength(s):{{filelength(s)}}</div>
      <div class="hit-fps">fps:{{fps}}</div>
      <div class="hit-faceattr">facial attributes:{{face_attributes}}</div>
      </div>
`,
            empty: 'No result for <q>{{ query }}</q>'
        }
    }),
    _widgets.configure({
        facets: [
            '*'
        ],
        maxValuesPerFacet: 20
    }),
    _widgets.pagination({
        container: '#pagination',
        showFirst: true,
        showLast: true
    }), 
]);
//SORT BY
search.addWidgets([
    _widgets.sortBy({
        container: '#sort-by',
        items: [
            {
                value: 'milestone1',
                label: 'frame (low - high)'
            },
            {
                value: 'milestone1_standard_filename_asce',
                label: 'filename (A - Z)'
            },
            {
                value: 'milestone1_standard_facescore_desc',
                label: 'face score (high - low)'
            },
            {
                value: 'milestone1_standard_actionscore_desc',
                label: 'action score (high - low)'
            }
        ]
    })
]);
// Set the InstantSearch index UI state from external events.
function setInstantSearchUiState(indexUiState) {
    search.setUiState((uiState)=>({
            ...uiState,
            [INSTANT_SEARCH_INDEX_NAME]: {
                ...uiState[INSTANT_SEARCH_INDEX_NAME],
                // We reset the page when the search state changes.
                page: 1,
                ...indexUiState
            }
        })
    );
}
function debounce(fn, time) {
    let timerId = undefined;
    return function(...args) {
        if (timerId) clearTimeout(timerId);
        timerId = setTimeout(()=>fn(...args)
        , time);
    };
}
const debouncedSetInstantSearchUiState = debounce(setInstantSearchUiState, 500);
const searchPageState = getInstantSearchUiState();
// Build URLs that InstantSearch understands.
function getInstantSearchUrl(indexUiState) {
    return search.createURL({
        [INSTANT_SEARCH_INDEX_NAME]: indexUiState
    });
}
// Return the InstantSearch index UI state.
function getInstantSearchUiState() {
    const uiState = instantSearchRouter.read();
    return uiState && uiState[INSTANT_SEARCH_INDEX_NAME] || {};
}
search.start();
// Detect when an event is modified with a special key to let the browser
// trigger its default behavior.
function isModifierEvent(event) {
    const isMiddleClick = event.button === 1;
    return isMiddleClick || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey;
}
function onSelect({ setIsOpen , setQuery , event , query  }) {
    // You want to trigger the default browser behavior if the event is modified.
    if (isModifierEvent(event)) return;
    setQuery(query);
    setIsOpen(false);
    setInstantSearchUiState({
        query
    });
}
function getItemUrl({ query  }) {
    return getInstantSearchUrl({
        query
    });
}
function createItemWrapperTemplate({ children , query , html  }) {
    const uiState = {
        query
    };
    return html`<a
    class="aa-ItemLink"
    href="${getInstantSearchUrl(uiState)}"
    onClick="${(event)=>{
        if (!isModifierEvent(event)) // Bypass the original link behavior if there's no event modifier
        // to set the InstantSearch UI state without reloading the page.
        event.preventDefault();
    }}"
  >
    ${children}
  </a>`;
}
const recentSearchesPlugin = _autocompletePluginRecentSearches.createLocalStorageRecentSearchesPlugin({
    key: 'instantsearch',
    limit: 5,
    transformSource ({ source  }) {
        return {
            ...source,
            getItemUrl ({ item  }) {
                return getItemUrl({
                    query: item.label
                });
            },
            onSelect ({ setIsOpen , setQuery , item , event  }) {
                onSelect({
                    setQuery,
                    setIsOpen,
                    event,
                    query: item.label
                });
            },
            // Update the default `item` template to wrap it with a link
            // and plug it to the InstantSearch router.
            templates: {
                ...source.templates,
                item (params) {
                    const { children  } = source.templates.item(params).props;
                    return createItemWrapperTemplate({
                        query: params.item.label,
                        children,
                        html: params.html
                    });
                }
            }
        };
    }
});
const querySuggestionsPlugin = _autocompletePluginQuerySuggestions.createQuerySuggestionsPlugin({
    searchClient,
    indexName: 'milestone1_query_suggestions3',
    getSearchParams ({ state  }) {
        return {
            hitsPerPage: state.query ? 5 : 10
        };
    // This creates a shared `hitsPerPage` value once the duplicates
    // between recent searches and Query Suggestions are removed.
    // return recentSearchesPlugin.data.getAlgoliaSearchParams({
    //  hitsPerPage: 6,
    // });
    },
    transformSource ({ source  }) {
        return {
            ...source,
            sourceId: 'querySuggestionsPlugin',
            getItemUrl ({ item  }) {
                return getItemUrl({
                    query: item.query
                });
            },
            onSelect ({ setIsOpen , setQuery , event , item  }) {
                onSelect({
                    setQuery,
                    setIsOpen,
                    event,
                    query: item.query
                });
            },
            getItems (params) {
                return source.getItems(params);
            },
            templates: {
                ...source.templates,
                item (params) {
                    const { children  } = source.templates.item(params).props;
                    return createItemWrapperTemplate({
                        query: params.item.label,
                        children,
                        html: params.html
                    });
                }
            }
        };
    }
});
_autocompleteJs.autocomplete({
    // You want recent searches to appear with an empty query.
    openOnFocus: true,
    // Add the recent searches and Query Suggestions plugins.
    plugins: [
        recentSearchesPlugin,
        querySuggestionsPlugin
    ],
    container: '#autocomplete',
    placeholder: 'Search for video..',
    detachedMediaQuery: 'none',
    initialState: {
        query: searchPageState.query || ''
    },
    onSubmit ({ state  }) {
        setInstantSearchUiState({
            query: state.query
        });
    },
    onReset () {
        setInstantSearchUiState({
            query: ''
        });
    },
    onStateChange ({ prevState , state  }) {
        if (prevState.query !== state.query) setInstantSearchUiState({
            query: state.query
        });
    }
});

},{"instantsearch.js":"5B89y","algoliasearch/lite":"ehDkI","@algolia/autocomplete-js":"3Syxs","instantsearch.js/es/connectors":"fWJNO","instantsearch.js/es/lib/routers/history":"haLSt","@algolia/autocomplete-plugin-query-suggestions":"kDyli","@algolia/autocomplete-plugin-recent-searches":"lFtzN","preact":"26zcy","instantsearch.js/es/widgets":"bk5Jd","@algolia/autocomplete-theme-classic":"1MBAZ","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"5B89y":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _instantSearch = require("./lib/InstantSearch");
var _instantSearchDefault = parcelHelpers.interopDefault(_instantSearch);
var _version = require("./lib/version");
var _versionDefault = parcelHelpers.interopDefault(_version);
var _helpers = require("./helpers");
var _infiniteHitsCache = require("./lib/infiniteHitsCache");
var instantsearch = function instantsearch(options) {
    return new _instantSearchDefault.default(options);
};
instantsearch.version = _versionDefault.default;
instantsearch.snippet = _helpers.snippet;
instantsearch.reverseSnippet = _helpers.reverseSnippet;
instantsearch.highlight = _helpers.highlight;
instantsearch.reverseHighlight = _helpers.reverseHighlight;
instantsearch.insights = _helpers.insights;
instantsearch.getInsightsAnonymousUserToken = _helpers.getInsightsAnonymousUserToken;
instantsearch.createInfiniteHitsSessionStorageCache = _infiniteHitsCache.createInfiniteHitsSessionStorageCache;
Object.defineProperty(instantsearch, 'widgets', {
    get: function get() {
        throw new ReferenceError("\"instantsearch.widgets\" are not available from the ES build.\n\nTo import the widgets:\n\nimport { searchBox } from 'instantsearch.js/es/widgets'");
    }
});
Object.defineProperty(instantsearch, 'connectors', {
    get: function get() {
        throw new ReferenceError("\"instantsearch.connectors\" are not available from the ES build.\n\nTo import the connectors:\n\nimport { connectSearchBox } from 'instantsearch.js/es/connectors'");
    }
});
exports.default = instantsearch;

},{"./lib/InstantSearch":"8mJmb","./lib/version":"hkkLK","./helpers":"8kgzi","./lib/infiniteHitsCache":"co24K","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"8mJmb":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _algoliasearchHelper = require("algoliasearch-helper");
var _algoliasearchHelperDefault = parcelHelpers.interopDefault(_algoliasearchHelper);
var _events = require("events");
var _eventsDefault = parcelHelpers.interopDefault(_events);
var _index = require("../widgets/index/index");
var _indexDefault = parcelHelpers.interopDefault(_index);
var _version = require("./version");
var _versionDefault = parcelHelpers.interopDefault(_version);
var _createHelpers = require("./createHelpers");
var _createHelpersDefault = parcelHelpers.interopDefault(_createHelpers);
var _utils = require("./utils");
var _createRouterMiddleware = require("../middlewares/createRouterMiddleware");
var _createMetadataMiddleware = require("../middlewares/createMetadataMiddleware");
function _typeof(obj1) {
    if (typeof Symbol === "function" && typeof Symbol.iterator === "symbol") _typeof = function _typeof(obj) {
        return typeof obj;
    };
    else _typeof = function _typeof(obj) {
        return obj && typeof Symbol === "function" && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj;
    };
    return _typeof(obj1);
}
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        if (enumerableOnly) symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        });
        keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = arguments[i] != null ? arguments[i] : {};
        if (i % 2) ownKeys(Object(source), true).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        });
        else if (Object.getOwnPropertyDescriptors) Object.defineProperties(target, Object.getOwnPropertyDescriptors(source));
        else ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _classCallCheck(instance, Constructor) {
    if (!(instance instanceof Constructor)) throw new TypeError("Cannot call a class as a function");
}
function _defineProperties(target, props) {
    for(var i = 0; i < props.length; i++){
        var descriptor = props[i];
        descriptor.enumerable = descriptor.enumerable || false;
        descriptor.configurable = true;
        if ("value" in descriptor) descriptor.writable = true;
        Object.defineProperty(target, descriptor.key, descriptor);
    }
}
function _createClass(Constructor, protoProps, staticProps) {
    if (protoProps) _defineProperties(Constructor.prototype, protoProps);
    if (staticProps) _defineProperties(Constructor, staticProps);
    return Constructor;
}
function _inherits(subClass, superClass) {
    if (typeof superClass !== "function" && superClass !== null) throw new TypeError("Super expression must either be null or a function");
    subClass.prototype = Object.create(superClass && superClass.prototype, {
        constructor: {
            value: subClass,
            writable: true,
            configurable: true
        }
    });
    if (superClass) _setPrototypeOf(subClass, superClass);
}
function _setPrototypeOf(o1, p1) {
    _setPrototypeOf = Object.setPrototypeOf || function _setPrototypeOf(o, p) {
        o.__proto__ = p;
        return o;
    };
    return _setPrototypeOf(o1, p1);
}
function _createSuper(Derived) {
    var hasNativeReflectConstruct = _isNativeReflectConstruct();
    return function _createSuperInternal() {
        var Super = _getPrototypeOf(Derived), result;
        if (hasNativeReflectConstruct) {
            var NewTarget = _getPrototypeOf(this).constructor;
            result = Reflect.construct(Super, arguments, NewTarget);
        } else result = Super.apply(this, arguments);
        return _possibleConstructorReturn(this, result);
    };
}
function _possibleConstructorReturn(self, call) {
    if (call && (_typeof(call) === "object" || typeof call === "function")) return call;
    return _assertThisInitialized(self);
}
function _assertThisInitialized(self) {
    if (self === void 0) throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
    return self;
}
function _isNativeReflectConstruct() {
    if (typeof Reflect === "undefined" || !Reflect.construct) return false;
    if (Reflect.construct.sham) return false;
    if (typeof Proxy === "function") return true;
    try {
        Date.prototype.toString.call(Reflect.construct(Date, [], function() {}));
        return true;
    } catch (e) {
        return false;
    }
}
function _getPrototypeOf(o2) {
    _getPrototypeOf = Object.setPrototypeOf ? Object.getPrototypeOf : function _getPrototypeOf(o) {
        return o.__proto__ || Object.getPrototypeOf(o);
    };
    return _getPrototypeOf(o2);
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
var withUsage = _utils.createDocumentationMessageGenerator({
    name: 'instantsearch'
});
function defaultCreateURL() {
    return '#';
}
/**
 * Global options for an InstantSearch instance.
 */ /**
 * The actual implementation of the InstantSearch. This is
 * created using the `instantsearch` factory function.
 * It emits the 'render' event every time a search is done
 */ var InstantSearch = /*#__PURE__*/ function(_EventEmitter) {
    _inherits(InstantSearch1, _EventEmitter);
    var _super = _createSuper(InstantSearch1);
    function InstantSearch1(options) {
        var _this;
        _classCallCheck(this, InstantSearch1);
        _this = _super.call(this);
        _defineProperty(_assertThisInitialized(_this), "client", void 0);
        _defineProperty(_assertThisInitialized(_this), "indexName", void 0);
        _defineProperty(_assertThisInitialized(_this), "insightsClient", void 0);
        _defineProperty(_assertThisInitialized(_this), "onStateChange", null);
        _defineProperty(_assertThisInitialized(_this), "helper", void 0);
        _defineProperty(_assertThisInitialized(_this), "mainHelper", void 0);
        _defineProperty(_assertThisInitialized(_this), "mainIndex", void 0);
        _defineProperty(_assertThisInitialized(_this), "started", void 0);
        _defineProperty(_assertThisInitialized(_this), "templatesConfig", void 0);
        _defineProperty(_assertThisInitialized(_this), "renderState", {});
        _defineProperty(_assertThisInitialized(_this), "_stalledSearchDelay", void 0);
        _defineProperty(_assertThisInitialized(_this), "_searchStalledTimer", void 0);
        _defineProperty(_assertThisInitialized(_this), "_isSearchStalled", void 0);
        _defineProperty(_assertThisInitialized(_this), "_initialUiState", void 0);
        _defineProperty(_assertThisInitialized(_this), "_createURL", void 0);
        _defineProperty(_assertThisInitialized(_this), "_searchFunction", void 0);
        _defineProperty(_assertThisInitialized(_this), "_mainHelperSearch", void 0);
        _defineProperty(_assertThisInitialized(_this), "middleware", []);
        _defineProperty(_assertThisInitialized(_this), "sendEventToInsights", void 0);
        _defineProperty(_assertThisInitialized(_this), "scheduleSearch", _utils.defer(function() {
            if (_this.started) _this.mainHelper.search();
        }));
        _defineProperty(_assertThisInitialized(_this), "scheduleRender", _utils.defer(function() {
            if (!_this.mainHelper.hasPendingRequests()) {
                clearTimeout(_this._searchStalledTimer);
                _this._searchStalledTimer = null;
                _this._isSearchStalled = false;
            }
            _this.mainIndex.render({
                instantSearchInstance: _assertThisInitialized(_this)
            });
            _this.emit('render');
        }));
        _defineProperty(_assertThisInitialized(_this), "onInternalStateChange", _utils.defer(function() {
            var nextUiState = _this.mainIndex.getWidgetUiState({});
            _this.middleware.forEach(function(_ref) {
                var instance = _ref.instance;
                instance.onStateChange({
                    uiState: nextUiState
                });
            });
        }));
        var _options$indexName = options.indexName, indexName = _options$indexName === void 0 ? null : _options$indexName, numberLocale = options.numberLocale, _options$initialUiSta = options.initialUiState, initialUiState = _options$initialUiSta === void 0 ? {} : _options$initialUiSta, _options$routing = options.routing, routing = _options$routing === void 0 ? null : _options$routing, searchFunction = options.searchFunction, _options$stalledSearc = options.stalledSearchDelay, stalledSearchDelay = _options$stalledSearc === void 0 ? 200 : _options$stalledSearc, _options$searchClient = options.searchClient, searchClient = _options$searchClient === void 0 ? null : _options$searchClient, _options$insightsClie = options.insightsClient, insightsClient = _options$insightsClie === void 0 ? null : _options$insightsClie, _options$onStateChang = options.onStateChange, onStateChange = _options$onStateChang === void 0 ? null : _options$onStateChang;
        if (indexName === null) throw new Error(withUsage('The `indexName` option is required.'));
        if (searchClient === null) throw new Error(withUsage('The `searchClient` option is required.'));
        if (typeof searchClient.search !== 'function') throw new Error("The `searchClient` must implement a `search` method.\n\nSee: https://www.algolia.com/doc/guides/building-search-ui/going-further/backend-search/in-depth/backend-instantsearch/js/");
        if (typeof searchClient.addAlgoliaAgent === 'function') searchClient.addAlgoliaAgent("instantsearch.js (".concat(_versionDefault.default, ")"));
        _utils.warning(insightsClient === null, "`insightsClient` property has been deprecated. It is still supported in 4.x releases, but not further. It is replaced by the `insights` middleware.\n\nFor more information, visit https://www.algolia.com/doc/guides/getting-insights-and-analytics/search-analytics/click-through-and-conversions/how-to/send-click-and-conversion-events-with-instantsearch/js/");
        if (insightsClient && typeof insightsClient !== 'function') throw new Error(withUsage('The `insightsClient` option should be a function.'));
        _utils.warning(!options.searchParameters, "The `searchParameters` option is deprecated and will not be supported in InstantSearch.js 4.x.\n\nYou can replace it with the `configure` widget:\n\n```\nsearch.addWidgets([\n  configure(".concat(JSON.stringify(options.searchParameters, null, 2), ")\n]);\n```\n\nSee ").concat(_utils.createDocumentationLink({
            name: 'configure'
        })));
        _this.client = searchClient;
        _this.insightsClient = insightsClient;
        _this.indexName = indexName;
        _this.helper = null;
        _this.mainHelper = null;
        _this.mainIndex = _indexDefault.default({
            indexName: indexName
        });
        _this.onStateChange = onStateChange;
        _this.started = false;
        _this.templatesConfig = {
            helpers: _createHelpersDefault.default({
                numberLocale: numberLocale
            }),
            compileOptions: {}
        };
        _this._stalledSearchDelay = stalledSearchDelay;
        _this._searchStalledTimer = null;
        _this._isSearchStalled = false;
        _this._createURL = defaultCreateURL;
        _this._initialUiState = initialUiState;
        if (searchFunction) _this._searchFunction = searchFunction;
        _this.sendEventToInsights = _utils.noop;
        if (routing) {
            var routerOptions = typeof routing === 'boolean' ? undefined : routing;
            _this.use(_createRouterMiddleware.createRouterMiddleware(routerOptions));
        }
        if (_createMetadataMiddleware.isMetadataEnabled()) _this.use(_createMetadataMiddleware.createMetadataMiddleware());
        return _this;
    }
    /**
   * Hooks a middleware into the InstantSearch lifecycle.
   */ _createClass(InstantSearch1, [
        {
            key: "use",
            value: function use() {
                var _this2 = this;
                for(var _len = arguments.length, middleware = new Array(_len), _key = 0; _key < _len; _key++)middleware[_key] = arguments[_key];
                var newMiddlewareList = middleware.map(function(fn) {
                    var newMiddleware = _objectSpread({
                        subscribe: _utils.noop,
                        unsubscribe: _utils.noop,
                        onStateChange: _utils.noop
                    }, fn({
                        instantSearchInstance: _this2
                    }));
                    _this2.middleware.push({
                        creator: fn,
                        instance: newMiddleware
                    });
                    return newMiddleware;
                }); // If the instance has already started, we directly subscribe the
                // middleware so they're notified of changes.
                if (this.started) newMiddlewareList.forEach(function(m) {
                    m.subscribe();
                });
                return this;
            }
        },
        {
            key: "unuse",
            value: function unuse() {
                for(var _len2 = arguments.length, middlewareToUnuse = new Array(_len2), _key2 = 0; _key2 < _len2; _key2++)middlewareToUnuse[_key2] = arguments[_key2];
                this.middleware.filter(function(m) {
                    return middlewareToUnuse.includes(m.creator);
                }).forEach(function(m) {
                    return m.instance.unsubscribe();
                });
                this.middleware = this.middleware.filter(function(m) {
                    return !middlewareToUnuse.includes(m.creator);
                });
                return this;
            } // @major we shipped with EXPERIMENTAL_use, but have changed that to just `use` now
        },
        {
            key: "EXPERIMENTAL_use",
            value: function EXPERIMENTAL_use() {
                _utils.warning(false, 'The middleware API is now considered stable, so we recommend replacing `EXPERIMENTAL_use` with `use` before upgrading to the next major version.');
                return this.use.apply(this, arguments);
            }
        },
        {
            key: "addWidget",
            value: function addWidget(widget) {
                _utils.warning(false, 'addWidget will still be supported in 4.x releases, but not further. It is replaced by `addWidgets([widget])`');
                return this.addWidgets([
                    widget
                ]);
            }
        },
        {
            key: "addWidgets",
            value: function addWidgets(widgets) {
                if (!Array.isArray(widgets)) throw new Error(withUsage('The `addWidgets` method expects an array of widgets. Please use `addWidget`.'));
                if (widgets.some(function(widget) {
                    return typeof widget.init !== 'function' && typeof widget.render !== 'function';
                })) throw new Error(withUsage('The widget definition expects a `render` and/or an `init` method.'));
                this.mainIndex.addWidgets(widgets);
                return this;
            }
        },
        {
            key: "removeWidget",
            value: function removeWidget(widget) {
                _utils.warning(false, 'removeWidget will still be supported in 4.x releases, but not further. It is replaced by `removeWidgets([widget])`');
                return this.removeWidgets([
                    widget
                ]);
            }
        },
        {
            key: "removeWidgets",
            value: function removeWidgets(widgets) {
                if (!Array.isArray(widgets)) throw new Error(withUsage('The `removeWidgets` method expects an array of widgets. Please use `removeWidget`.'));
                if (widgets.some(function(widget) {
                    return typeof widget.dispose !== 'function';
                })) throw new Error(withUsage('The widget definition expects a `dispose` method.'));
                this.mainIndex.removeWidgets(widgets);
                return this;
            }
        },
        {
            key: "start",
            value: function start() {
                var _this3 = this;
                if (this.started) throw new Error(withUsage('The `start` method has already been called once.'));
                 // This Helper is used for the queries, we don't care about its state. The
                // states are managed at the `index` level. We use this Helper to create
                // DerivedHelper scoped into the `index` widgets.
                var mainHelper = _algoliasearchHelperDefault.default(this.client, this.indexName);
                mainHelper.search = function() {
                    // This solution allows us to keep the exact same API for the users but
                    // under the hood, we have a different implementation. It should be
                    // completely transparent for the rest of the codebase. Only this module
                    // is impacted.
                    return mainHelper.searchOnlyWithDerivedHelpers();
                };
                if (this._searchFunction) {
                    // this client isn't used to actually search, but required for the helper
                    // to not throw errors
                    var fakeClient = {
                        search: function search() {
                            return new Promise(_utils.noop);
                        }
                    };
                    this._mainHelperSearch = mainHelper.search.bind(mainHelper);
                    mainHelper.search = function() {
                        var mainIndexHelper = _this3.mainIndex.getHelper();
                        var searchFunctionHelper = _algoliasearchHelperDefault.default(fakeClient, mainIndexHelper.state.index, mainIndexHelper.state);
                        searchFunctionHelper.once('search', function(_ref2) {
                            var state = _ref2.state;
                            mainIndexHelper.overrideStateWithoutTriggeringChangeEvent(state);
                            _this3._mainHelperSearch();
                        }); // Forward state changes from `searchFunctionHelper` to `mainIndexHelper`
                        searchFunctionHelper.on('change', function(_ref3) {
                            var state = _ref3.state;
                            mainIndexHelper.setState(state);
                        });
                        _this3._searchFunction(searchFunctionHelper);
                        return mainHelper;
                    };
                } // Only the "main" Helper emits the `error` event vs the one for `search`
                // and `results` that are also emitted on the derived one.
                mainHelper.on('error', function(_ref4) {
                    var error = _ref4.error;
                    _this3.emit('error', {
                        error: error
                    });
                });
                this.mainHelper = mainHelper;
                this.mainIndex.init({
                    instantSearchInstance: this,
                    parent: null,
                    uiState: this._initialUiState
                });
                this.middleware.forEach(function(_ref5) {
                    var instance = _ref5.instance;
                    instance.subscribe();
                });
                mainHelper.search(); // Keep the previous reference for legacy purpose, some pattern use
                // the direct Helper access `search.helper` (e.g multi-index).
                this.helper = this.mainIndex.getHelper(); // track we started the search if we add more widgets,
                // to init them directly after add
                this.started = true;
            }
        },
        {
            key: "dispose",
            value: function dispose() {
                this.scheduleSearch.cancel();
                this.scheduleRender.cancel();
                clearTimeout(this._searchStalledTimer);
                this.removeWidgets(this.mainIndex.getWidgets());
                this.mainIndex.dispose(); // You can not start an instance two times, therefore a disposed instance
                // needs to set started as false otherwise this can not be restarted at a
                // later point.
                this.started = false; // The helper needs to be reset to perform the next search from a fresh state.
                // If not reset, it would use the state stored before calling `dispose()`.
                this.removeAllListeners();
                this.mainHelper.removeAllListeners();
                this.mainHelper = null;
                this.helper = null;
                this.middleware.forEach(function(_ref6) {
                    var instance = _ref6.instance;
                    instance.unsubscribe();
                });
            }
        },
        {
            key: "scheduleStalledRender",
            value: function scheduleStalledRender() {
                var _this4 = this;
                if (!this._searchStalledTimer) this._searchStalledTimer = setTimeout(function() {
                    _this4._isSearchStalled = true;
                    _this4.scheduleRender();
                }, this._stalledSearchDelay);
            }
        },
        {
            key: "setUiState",
            value: function setUiState(uiState) {
                if (!this.mainHelper) throw new Error(withUsage('The `start` method needs to be called before `setUiState`.'));
                 // We refresh the index UI state to update the local UI state that the
                // main index passes to the function form of `setUiState`.
                this.mainIndex.refreshUiState();
                var nextUiState = typeof uiState === 'function' ? uiState(this.mainIndex.getWidgetUiState({})) : uiState;
                var setIndexHelperState1 = function setIndexHelperState(indexWidget) {
                    _utils.checkIndexUiState({
                        index: indexWidget,
                        indexUiState: nextUiState[indexWidget.getIndexId()]
                    });
                    indexWidget.getHelper().setState(indexWidget.getWidgetSearchParameters(indexWidget.getHelper().state, {
                        uiState: nextUiState[indexWidget.getIndexId()]
                    }));
                    indexWidget.getWidgets().filter(_index.isIndexWidget).forEach(setIndexHelperState);
                };
                setIndexHelperState1(this.mainIndex);
                this.scheduleSearch();
            }
        },
        {
            key: "getUiState",
            value: function getUiState() {
                if (this.started) // We refresh the index UI state to make sure changes from `refine` are taken in account
                this.mainIndex.refreshUiState();
                return this.mainIndex.getWidgetUiState({});
            }
        },
        {
            key: "createURL",
            value: function createURL() {
                var nextState = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : {};
                if (!this.started) throw new Error(withUsage('The `start` method needs to be called before `createURL`.'));
                return this._createURL(nextState);
            }
        },
        {
            key: "refresh",
            value: function refresh() {
                if (!this.mainHelper) throw new Error(withUsage('The `start` method needs to be called before `refresh`.'));
                this.mainHelper.clearCache().search();
            }
        }
    ]);
    return InstantSearch1;
}(_eventsDefault.default);
exports.default = InstantSearch;

},{"algoliasearch-helper":"jGqjt","events":"1VQLm","../widgets/index/index":"kdZTz","./version":"hkkLK","./createHelpers":"8IHo3","./utils":"etVYs","../middlewares/createRouterMiddleware":"4mKEu","../middlewares/createMetadataMiddleware":"lOLJd","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"jGqjt":[function(require,module,exports) {
'use strict';
var AlgoliaSearchHelper = require('./src/algoliasearch.helper');
var SearchParameters = require('./src/SearchParameters');
var SearchResults = require('./src/SearchResults');
/**
 * The algoliasearchHelper module is the function that will let its
 * contains everything needed to use the Algoliasearch
 * Helper. It is a also a function that instanciate the helper.
 * To use the helper, you also need the Algolia JS client v3.
 * @example
 * //using the UMD build
 * var client = algoliasearch('latency', '6be0576ff61c053d5f9a3225e2a90f76');
 * var helper = algoliasearchHelper(client, 'bestbuy', {
 *   facets: ['shipping'],
 *   disjunctiveFacets: ['category']
 * });
 * helper.on('result', function(event) {
 *   console.log(event.results);
 * });
 * helper
 *   .toggleFacetRefinement('category', 'Movies & TV Shows')
 *   .toggleFacetRefinement('shipping', 'Free shipping')
 *   .search();
 * @example
 * // The helper is an event emitter using the node API
 * helper.on('result', updateTheResults);
 * helper.once('result', updateTheResults);
 * helper.removeListener('result', updateTheResults);
 * helper.removeAllListeners('result');
 * @module algoliasearchHelper
 * @param  {AlgoliaSearch} client an AlgoliaSearch client
 * @param  {string} index the name of the index to query
 * @param  {SearchParameters|object} opts an object defining the initial config of the search. It doesn't have to be a {SearchParameters}, just an object containing the properties you need from it.
 * @return {AlgoliaSearchHelper}
 */ function algoliasearchHelper(client, index, opts) {
    return new AlgoliaSearchHelper(client, index, opts);
}
/**
 * The version currently used
 * @member module:algoliasearchHelper.version
 * @type {number}
 */ algoliasearchHelper.version = require('./src/version.js');
/**
 * Constructor for the Helper.
 * @member module:algoliasearchHelper.AlgoliaSearchHelper
 * @type {AlgoliaSearchHelper}
 */ algoliasearchHelper.AlgoliaSearchHelper = AlgoliaSearchHelper;
/**
 * Constructor for the object containing all the parameters of the search.
 * @member module:algoliasearchHelper.SearchParameters
 * @type {SearchParameters}
 */ algoliasearchHelper.SearchParameters = SearchParameters;
/**
 * Constructor for the object containing the results of the search.
 * @member module:algoliasearchHelper.SearchResults
 * @type {SearchResults}
 */ algoliasearchHelper.SearchResults = SearchResults;
module.exports = algoliasearchHelper;

},{"./src/algoliasearch.helper":"jewxp","./src/SearchParameters":"dQfwH","./src/SearchResults":"lUGU6","./src/version.js":"cs17k"}],"jewxp":[function(require,module,exports) {
'use strict';
var SearchParameters = require('./SearchParameters');
var SearchResults = require('./SearchResults');
var DerivedHelper = require('./DerivedHelper');
var requestBuilder = require('./requestBuilder');
var EventEmitter = require('@algolia/events');
var inherits = require('./functions/inherits');
var objectHasKeys = require('./functions/objectHasKeys');
var omit = require('./functions/omit');
var merge = require('./functions/merge');
var version = require('./version');
var escapeFacetValue = require('./functions/escapeFacetValue').escapeFacetValue;
/**
 * Event triggered when a parameter is set or updated
 * @event AlgoliaSearchHelper#event:change
 * @property {object} event
 * @property {SearchParameters} event.state the current parameters with the latest changes applied
 * @property {SearchResults} event.results the previous results received from Algolia. `null` before the first request
 * @example
 * helper.on('change', function(event) {
 *   console.log('The parameters have changed');
 * });
 */ /**
 * Event triggered when a main search is sent to Algolia
 * @event AlgoliaSearchHelper#event:search
 * @property {object} event
 * @property {SearchParameters} event.state the parameters used for this search
 * @property {SearchResults} event.results the results from the previous search. `null` if it is the first search.
 * @example
 * helper.on('search', function(event) {
 *   console.log('Search sent');
 * });
 */ /**
 * Event triggered when a search using `searchForFacetValues` is sent to Algolia
 * @event AlgoliaSearchHelper#event:searchForFacetValues
 * @property {object} event
 * @property {SearchParameters} event.state the parameters used for this search it is the first search.
 * @property {string} event.facet the facet searched into
 * @property {string} event.query the query used to search in the facets
 * @example
 * helper.on('searchForFacetValues', function(event) {
 *   console.log('searchForFacetValues sent');
 * });
 */ /**
 * Event triggered when a search using `searchOnce` is sent to Algolia
 * @event AlgoliaSearchHelper#event:searchOnce
 * @property {object} event
 * @property {SearchParameters} event.state the parameters used for this search it is the first search.
 * @example
 * helper.on('searchOnce', function(event) {
 *   console.log('searchOnce sent');
 * });
 */ /**
 * Event triggered when the results are retrieved from Algolia
 * @event AlgoliaSearchHelper#event:result
 * @property {object} event
 * @property {SearchResults} event.results the results received from Algolia
 * @property {SearchParameters} event.state the parameters used to query Algolia. Those might be different from the one in the helper instance (for example if the network is unreliable).
 * @example
 * helper.on('result', function(event) {
 *   console.log('Search results received');
 * });
 */ /**
 * Event triggered when Algolia sends back an error. For example, if an unknown parameter is
 * used, the error can be caught using this event.
 * @event AlgoliaSearchHelper#event:error
 * @property {object} event
 * @property {Error} event.error the error returned by the Algolia.
 * @example
 * helper.on('error', function(event) {
 *   console.log('Houston we got a problem.');
 * });
 */ /**
 * Event triggered when the queue of queries have been depleted (with any result or outdated queries)
 * @event AlgoliaSearchHelper#event:searchQueueEmpty
 * @example
 * helper.on('searchQueueEmpty', function() {
 *   console.log('No more search pending');
 *   // This is received before the result event if we're not expecting new results
 * });
 *
 * helper.search();
 */ /**
 * Initialize a new AlgoliaSearchHelper
 * @class
 * @classdesc The AlgoliaSearchHelper is a class that ease the management of the
 * search. It provides an event based interface for search callbacks:
 *  - change: when the internal search state is changed.
 *    This event contains a {@link SearchParameters} object and the
 *    {@link SearchResults} of the last result if any.
 *  - search: when a search is triggered using the `search()` method.
 *  - result: when the response is retrieved from Algolia and is processed.
 *    This event contains a {@link SearchResults} object and the
 *    {@link SearchParameters} corresponding to this answer.
 *  - error: when the response is an error. This event contains the error returned by the server.
 * @param  {AlgoliaSearch} client an AlgoliaSearch client
 * @param  {string} index the index name to query
 * @param  {SearchParameters | object} options an object defining the initial
 * config of the search. It doesn't have to be a {SearchParameters},
 * just an object containing the properties you need from it.
 */ function AlgoliaSearchHelper(client, index, options) {
    if (typeof client.addAlgoliaAgent === 'function') client.addAlgoliaAgent('JS Helper (' + version + ')');
    this.setClient(client);
    var opts = options || {};
    opts.index = index;
    this.state = SearchParameters.make(opts);
    this.lastResults = null;
    this._queryId = 0;
    this._lastQueryIdReceived = -1;
    this.derivedHelpers = [];
    this._currentNbQueries = 0;
}
inherits(AlgoliaSearchHelper, EventEmitter);
/**
 * Start the search with the parameters set in the state. When the
 * method is called, it triggers a `search` event. The results will
 * be available through the `result` event. If an error occurs, an
 * `error` will be fired instead.
 * @return {AlgoliaSearchHelper}
 * @fires search
 * @fires result
 * @fires error
 * @chainable
 */ AlgoliaSearchHelper.prototype.search = function() {
    this._search({
        onlyWithDerivedHelpers: false
    });
    return this;
};
AlgoliaSearchHelper.prototype.searchOnlyWithDerivedHelpers = function() {
    this._search({
        onlyWithDerivedHelpers: true
    });
    return this;
};
/**
 * Gets the search query parameters that would be sent to the Algolia Client
 * for the hits
 * @return {object} Query Parameters
 */ AlgoliaSearchHelper.prototype.getQuery = function() {
    var state = this.state;
    return requestBuilder._getHitsSearchParams(state);
};
/**
 * Start a search using a modified version of the current state. This method does
 * not trigger the helper lifecycle and does not modify the state kept internally
 * by the helper. This second aspect means that the next search call will be the
 * same as a search call before calling searchOnce.
 * @param {object} options can contain all the parameters that can be set to SearchParameters
 * plus the index
 * @param {function} [callback] optional callback executed when the response from the
 * server is back.
 * @return {promise|undefined} if a callback is passed the method returns undefined
 * otherwise it returns a promise containing an object with two keys :
 *  - content with a SearchResults
 *  - state with the state used for the query as a SearchParameters
 * @example
 * // Changing the number of records returned per page to 1
 * // This example uses the callback API
 * var state = helper.searchOnce({hitsPerPage: 1},
 *   function(error, content, state) {
 *     // if an error occurred it will be passed in error, otherwise its value is null
 *     // content contains the results formatted as a SearchResults
 *     // state is the instance of SearchParameters used for this search
 *   });
 * @example
 * // Changing the number of records returned per page to 1
 * // This example uses the promise API
 * var state1 = helper.searchOnce({hitsPerPage: 1})
 *                 .then(promiseHandler);
 *
 * function promiseHandler(res) {
 *   // res contains
 *   // {
 *   //   content : SearchResults
 *   //   state   : SearchParameters (the one used for this specific search)
 *   // }
 * }
 */ AlgoliaSearchHelper.prototype.searchOnce = function(options, cb) {
    var tempState = !options ? this.state : this.state.setQueryParameters(options);
    var queries = requestBuilder._getQueries(tempState.index, tempState);
    var self = this;
    this._currentNbQueries++;
    this.emit('searchOnce', {
        state: tempState
    });
    if (cb) {
        this.client.search(queries).then(function(content) {
            self._currentNbQueries--;
            if (self._currentNbQueries === 0) self.emit('searchQueueEmpty');
            cb(null, new SearchResults(tempState, content.results), tempState);
        }).catch(function(err) {
            self._currentNbQueries--;
            if (self._currentNbQueries === 0) self.emit('searchQueueEmpty');
            cb(err, null, tempState);
        });
        return undefined;
    }
    return this.client.search(queries).then(function(content) {
        self._currentNbQueries--;
        if (self._currentNbQueries === 0) self.emit('searchQueueEmpty');
        return {
            content: new SearchResults(tempState, content.results),
            state: tempState,
            _originalResponse: content
        };
    }, function(e) {
        self._currentNbQueries--;
        if (self._currentNbQueries === 0) self.emit('searchQueueEmpty');
        throw e;
    });
};
/**
 * Start the search for answers with the parameters set in the state.
 * This method returns a promise.
 * @param {Object} options - the options for answers API call
 * @param {string[]} options.attributesForPrediction - Attributes to use for predictions. If empty, `searchableAttributes` is used instead.
 * @param {string[]} options.queryLanguages - The languages in the query. Currently only supports ['en'].
 * @param {number} options.nbHits - Maximum number of answers to retrieve from the Answers Engine. Cannot be greater than 1000.
 *
 * @return {promise} the answer results
 */ AlgoliaSearchHelper.prototype.findAnswers = function(options) {
    var state = this.state;
    var derivedHelper = this.derivedHelpers[0];
    if (!derivedHelper) return Promise.resolve([]);
    var derivedState = derivedHelper.getModifiedState(state);
    var data = merge({
        attributesForPrediction: options.attributesForPrediction,
        nbHits: options.nbHits
    }, {
        params: omit(requestBuilder._getHitsSearchParams(derivedState), [
            'attributesToSnippet',
            'hitsPerPage',
            'restrictSearchableAttributes',
            'snippetEllipsisText' // FIXME remove this line once the engine is fixed.
        ])
    });
    var errorMessage = 'search for answers was called, but this client does not have a function client.initIndex(index).findAnswers';
    if (typeof this.client.initIndex !== 'function') throw new Error(errorMessage);
    var index = this.client.initIndex(derivedState.index);
    if (typeof index.findAnswers !== 'function') throw new Error(errorMessage);
    return index.findAnswers(derivedState.query, options.queryLanguages, data);
};
/**
 * Structure of each result when using
 * [`searchForFacetValues()`](reference.html#AlgoliaSearchHelper#searchForFacetValues)
 * @typedef FacetSearchHit
 * @type {object}
 * @property {string} value the facet value
 * @property {string} highlighted the facet value highlighted with the query string
 * @property {number} count number of occurrence of this facet value
 * @property {boolean} isRefined true if the value is already refined
 */ /**
 * Structure of the data resolved by the
 * [`searchForFacetValues()`](reference.html#AlgoliaSearchHelper#searchForFacetValues)
 * promise.
 * @typedef FacetSearchResult
 * @type {object}
 * @property {FacetSearchHit} facetHits the results for this search for facet values
 * @property {number} processingTimeMS time taken by the query inside the engine
 */ /**
 * Search for facet values based on an query and the name of a faceted attribute. This
 * triggers a search and will return a promise. On top of using the query, it also sends
 * the parameters from the state so that the search is narrowed down to only the possible values.
 *
 * See the description of [FacetSearchResult](reference.html#FacetSearchResult)
 * @param {string} facet the name of the faceted attribute
 * @param {string} query the string query for the search
 * @param {number} [maxFacetHits] the maximum number values returned. Should be > 0 and <= 100
 * @param {object} [userState] the set of custom parameters to use on top of the current state. Setting a property to `undefined` removes
 * it in the generated query.
 * @return {promise.<FacetSearchResult>} the results of the search
 */ AlgoliaSearchHelper.prototype.searchForFacetValues = function(facet, query, maxFacetHits, userState) {
    var clientHasSFFV = typeof this.client.searchForFacetValues === 'function';
    if (!clientHasSFFV && typeof this.client.initIndex !== 'function') throw new Error('search for facet values (searchable) was called, but this client does not have a function client.searchForFacetValues or client.initIndex(index).searchForFacetValues');
    var state = this.state.setQueryParameters(userState || {});
    var isDisjunctive = state.isDisjunctiveFacet(facet);
    var algoliaQuery = requestBuilder.getSearchForFacetQuery(facet, query, maxFacetHits, state);
    this._currentNbQueries++;
    var self = this;
    this.emit('searchForFacetValues', {
        state: state,
        facet: facet,
        query: query
    });
    var searchForFacetValuesPromise = clientHasSFFV ? this.client.searchForFacetValues([
        {
            indexName: state.index,
            params: algoliaQuery
        }
    ]) : this.client.initIndex(state.index).searchForFacetValues(algoliaQuery);
    return searchForFacetValuesPromise.then(function addIsRefined(content) {
        self._currentNbQueries--;
        if (self._currentNbQueries === 0) self.emit('searchQueueEmpty');
        content = Array.isArray(content) ? content[0] : content;
        content.facetHits.forEach(function(f) {
            f.escapedValue = escapeFacetValue(f.value);
            f.isRefined = isDisjunctive ? state.isDisjunctiveFacetRefined(facet, f.escapedValue) : state.isFacetRefined(facet, f.escapedValue);
        });
        return content;
    }, function(e) {
        self._currentNbQueries--;
        if (self._currentNbQueries === 0) self.emit('searchQueueEmpty');
        throw e;
    });
};
/**
 * Sets the text query used for the search.
 *
 * This method resets the current page to 0.
 * @param  {string} q the user query
 * @return {AlgoliaSearchHelper}
 * @fires change
 * @chainable
 */ AlgoliaSearchHelper.prototype.setQuery = function(q) {
    this._change({
        state: this.state.resetPage().setQuery(q),
        isPageReset: true
    });
    return this;
};
/**
 * Remove all the types of refinements except tags. A string can be provided to remove
 * only the refinements of a specific attribute. For more advanced use case, you can
 * provide a function instead. This function should follow the
 * [clearCallback definition](#SearchParameters.clearCallback).
 *
 * This method resets the current page to 0.
 * @param {string} [name] optional name of the facet / attribute on which we want to remove all refinements
 * @return {AlgoliaSearchHelper}
 * @fires change
 * @chainable
 * @example
 * // Removing all the refinements
 * helper.clearRefinements().search();
 * @example
 * // Removing all the filters on a the category attribute.
 * helper.clearRefinements('category').search();
 * @example
 * // Removing only the exclude filters on the category facet.
 * helper.clearRefinements(function(value, attribute, type) {
 *   return type === 'exclude' && attribute === 'category';
 * }).search();
 */ AlgoliaSearchHelper.prototype.clearRefinements = function(name) {
    this._change({
        state: this.state.resetPage().clearRefinements(name),
        isPageReset: true
    });
    return this;
};
/**
 * Remove all the tag filters.
 *
 * This method resets the current page to 0.
 * @return {AlgoliaSearchHelper}
 * @fires change
 * @chainable
 */ AlgoliaSearchHelper.prototype.clearTags = function() {
    this._change({
        state: this.state.resetPage().clearTags(),
        isPageReset: true
    });
    return this;
};
/**
 * Adds a disjunctive filter to a faceted attribute with the `value` provided. If the
 * filter is already set, it doesn't change the filters.
 *
 * This method resets the current page to 0.
 * @param  {string} facet the facet to refine
 * @param  {string} value the associated value (will be converted to string)
 * @return {AlgoliaSearchHelper}
 * @fires change
 * @chainable
 */ AlgoliaSearchHelper.prototype.addDisjunctiveFacetRefinement = function(facet, value) {
    this._change({
        state: this.state.resetPage().addDisjunctiveFacetRefinement(facet, value),
        isPageReset: true
    });
    return this;
};
/**
 * @deprecated since version 2.4.0, see {@link AlgoliaSearchHelper#addDisjunctiveFacetRefinement}
 */ AlgoliaSearchHelper.prototype.addDisjunctiveRefine = function() {
    return this.addDisjunctiveFacetRefinement.apply(this, arguments);
};
/**
 * Adds a refinement on a hierarchical facet. It will throw
 * an exception if the facet is not defined or if the facet
 * is already refined.
 *
 * This method resets the current page to 0.
 * @param {string} facet the facet name
 * @param {string} path the hierarchical facet path
 * @return {AlgoliaSearchHelper}
 * @throws Error if the facet is not defined or if the facet is refined
 * @chainable
 * @fires change
 */ AlgoliaSearchHelper.prototype.addHierarchicalFacetRefinement = function(facet, value) {
    this._change({
        state: this.state.resetPage().addHierarchicalFacetRefinement(facet, value),
        isPageReset: true
    });
    return this;
};
/**
 * Adds a an numeric filter to an attribute with the `operator` and `value` provided. If the
 * filter is already set, it doesn't change the filters.
 *
 * This method resets the current page to 0.
 * @param  {string} attribute the attribute on which the numeric filter applies
 * @param  {string} operator the operator of the filter
 * @param  {number} value the value of the filter
 * @return {AlgoliaSearchHelper}
 * @fires change
 * @chainable
 */ AlgoliaSearchHelper.prototype.addNumericRefinement = function(attribute, operator, value) {
    this._change({
        state: this.state.resetPage().addNumericRefinement(attribute, operator, value),
        isPageReset: true
    });
    return this;
};
/**
 * Adds a filter to a faceted attribute with the `value` provided. If the
 * filter is already set, it doesn't change the filters.
 *
 * This method resets the current page to 0.
 * @param  {string} facet the facet to refine
 * @param  {string} value the associated value (will be converted to string)
 * @return {AlgoliaSearchHelper}
 * @fires change
 * @chainable
 */ AlgoliaSearchHelper.prototype.addFacetRefinement = function(facet, value) {
    this._change({
        state: this.state.resetPage().addFacetRefinement(facet, value),
        isPageReset: true
    });
    return this;
};
/**
 * @deprecated since version 2.4.0, see {@link AlgoliaSearchHelper#addFacetRefinement}
 */ AlgoliaSearchHelper.prototype.addRefine = function() {
    return this.addFacetRefinement.apply(this, arguments);
};
/**
 * Adds a an exclusion filter to a faceted attribute with the `value` provided. If the
 * filter is already set, it doesn't change the filters.
 *
 * This method resets the current page to 0.
 * @param  {string} facet the facet to refine
 * @param  {string} value the associated value (will be converted to string)
 * @return {AlgoliaSearchHelper}
 * @fires change
 * @chainable
 */ AlgoliaSearchHelper.prototype.addFacetExclusion = function(facet, value) {
    this._change({
        state: this.state.resetPage().addExcludeRefinement(facet, value),
        isPageReset: true
    });
    return this;
};
/**
 * @deprecated since version 2.4.0, see {@link AlgoliaSearchHelper#addFacetExclusion}
 */ AlgoliaSearchHelper.prototype.addExclude = function() {
    return this.addFacetExclusion.apply(this, arguments);
};
/**
 * Adds a tag filter with the `tag` provided. If the
 * filter is already set, it doesn't change the filters.
 *
 * This method resets the current page to 0.
 * @param {string} tag the tag to add to the filter
 * @return {AlgoliaSearchHelper}
 * @fires change
 * @chainable
 */ AlgoliaSearchHelper.prototype.addTag = function(tag) {
    this._change({
        state: this.state.resetPage().addTagRefinement(tag),
        isPageReset: true
    });
    return this;
};
/**
 * Removes an numeric filter to an attribute with the `operator` and `value` provided. If the
 * filter is not set, it doesn't change the filters.
 *
 * Some parameters are optional, triggering different behavior:
 *  - if the value is not provided, then all the numeric value will be removed for the
 *  specified attribute/operator couple.
 *  - if the operator is not provided either, then all the numeric filter on this attribute
 *  will be removed.
 *
 * This method resets the current page to 0.
 * @param  {string} attribute the attribute on which the numeric filter applies
 * @param  {string} [operator] the operator of the filter
 * @param  {number} [value] the value of the filter
 * @return {AlgoliaSearchHelper}
 * @fires change
 * @chainable
 */ AlgoliaSearchHelper.prototype.removeNumericRefinement = function(attribute, operator, value) {
    this._change({
        state: this.state.resetPage().removeNumericRefinement(attribute, operator, value),
        isPageReset: true
    });
    return this;
};
/**
 * Removes a disjunctive filter to a faceted attribute with the `value` provided. If the
 * filter is not set, it doesn't change the filters.
 *
 * If the value is omitted, then this method will remove all the filters for the
 * attribute.
 *
 * This method resets the current page to 0.
 * @param  {string} facet the facet to refine
 * @param  {string} [value] the associated value
 * @return {AlgoliaSearchHelper}
 * @fires change
 * @chainable
 */ AlgoliaSearchHelper.prototype.removeDisjunctiveFacetRefinement = function(facet, value) {
    this._change({
        state: this.state.resetPage().removeDisjunctiveFacetRefinement(facet, value),
        isPageReset: true
    });
    return this;
};
/**
 * @deprecated since version 2.4.0, see {@link AlgoliaSearchHelper#removeDisjunctiveFacetRefinement}
 */ AlgoliaSearchHelper.prototype.removeDisjunctiveRefine = function() {
    return this.removeDisjunctiveFacetRefinement.apply(this, arguments);
};
/**
 * Removes the refinement set on a hierarchical facet.
 * @param {string} facet the facet name
 * @return {AlgoliaSearchHelper}
 * @throws Error if the facet is not defined or if the facet is not refined
 * @fires change
 * @chainable
 */ AlgoliaSearchHelper.prototype.removeHierarchicalFacetRefinement = function(facet) {
    this._change({
        state: this.state.resetPage().removeHierarchicalFacetRefinement(facet),
        isPageReset: true
    });
    return this;
};
/**
 * Removes a filter to a faceted attribute with the `value` provided. If the
 * filter is not set, it doesn't change the filters.
 *
 * If the value is omitted, then this method will remove all the filters for the
 * attribute.
 *
 * This method resets the current page to 0.
 * @param  {string} facet the facet to refine
 * @param  {string} [value] the associated value
 * @return {AlgoliaSearchHelper}
 * @fires change
 * @chainable
 */ AlgoliaSearchHelper.prototype.removeFacetRefinement = function(facet, value) {
    this._change({
        state: this.state.resetPage().removeFacetRefinement(facet, value),
        isPageReset: true
    });
    return this;
};
/**
 * @deprecated since version 2.4.0, see {@link AlgoliaSearchHelper#removeFacetRefinement}
 */ AlgoliaSearchHelper.prototype.removeRefine = function() {
    return this.removeFacetRefinement.apply(this, arguments);
};
/**
 * Removes an exclusion filter to a faceted attribute with the `value` provided. If the
 * filter is not set, it doesn't change the filters.
 *
 * If the value is omitted, then this method will remove all the filters for the
 * attribute.
 *
 * This method resets the current page to 0.
 * @param  {string} facet the facet to refine
 * @param  {string} [value] the associated value
 * @return {AlgoliaSearchHelper}
 * @fires change
 * @chainable
 */ AlgoliaSearchHelper.prototype.removeFacetExclusion = function(facet, value) {
    this._change({
        state: this.state.resetPage().removeExcludeRefinement(facet, value),
        isPageReset: true
    });
    return this;
};
/**
 * @deprecated since version 2.4.0, see {@link AlgoliaSearchHelper#removeFacetExclusion}
 */ AlgoliaSearchHelper.prototype.removeExclude = function() {
    return this.removeFacetExclusion.apply(this, arguments);
};
/**
 * Removes a tag filter with the `tag` provided. If the
 * filter is not set, it doesn't change the filters.
 *
 * This method resets the current page to 0.
 * @param {string} tag tag to remove from the filter
 * @return {AlgoliaSearchHelper}
 * @fires change
 * @chainable
 */ AlgoliaSearchHelper.prototype.removeTag = function(tag) {
    this._change({
        state: this.state.resetPage().removeTagRefinement(tag),
        isPageReset: true
    });
    return this;
};
/**
 * Adds or removes an exclusion filter to a faceted attribute with the `value` provided. If
 * the value is set then it removes it, otherwise it adds the filter.
 *
 * This method resets the current page to 0.
 * @param  {string} facet the facet to refine
 * @param  {string} value the associated value
 * @return {AlgoliaSearchHelper}
 * @fires change
 * @chainable
 */ AlgoliaSearchHelper.prototype.toggleFacetExclusion = function(facet, value) {
    this._change({
        state: this.state.resetPage().toggleExcludeFacetRefinement(facet, value),
        isPageReset: true
    });
    return this;
};
/**
 * @deprecated since version 2.4.0, see {@link AlgoliaSearchHelper#toggleFacetExclusion}
 */ AlgoliaSearchHelper.prototype.toggleExclude = function() {
    return this.toggleFacetExclusion.apply(this, arguments);
};
/**
 * Adds or removes a filter to a faceted attribute with the `value` provided. If
 * the value is set then it removes it, otherwise it adds the filter.
 *
 * This method can be used for conjunctive, disjunctive and hierarchical filters.
 *
 * This method resets the current page to 0.
 * @param  {string} facet the facet to refine
 * @param  {string} value the associated value
 * @return {AlgoliaSearchHelper}
 * @throws Error will throw an error if the facet is not declared in the settings of the helper
 * @fires change
 * @chainable
 * @deprecated since version 2.19.0, see {@link AlgoliaSearchHelper#toggleFacetRefinement}
 */ AlgoliaSearchHelper.prototype.toggleRefinement = function(facet, value) {
    return this.toggleFacetRefinement(facet, value);
};
/**
 * Adds or removes a filter to a faceted attribute with the `value` provided. If
 * the value is set then it removes it, otherwise it adds the filter.
 *
 * This method can be used for conjunctive, disjunctive and hierarchical filters.
 *
 * This method resets the current page to 0.
 * @param  {string} facet the facet to refine
 * @param  {string} value the associated value
 * @return {AlgoliaSearchHelper}
 * @throws Error will throw an error if the facet is not declared in the settings of the helper
 * @fires change
 * @chainable
 */ AlgoliaSearchHelper.prototype.toggleFacetRefinement = function(facet, value) {
    this._change({
        state: this.state.resetPage().toggleFacetRefinement(facet, value),
        isPageReset: true
    });
    return this;
};
/**
 * @deprecated since version 2.4.0, see {@link AlgoliaSearchHelper#toggleFacetRefinement}
 */ AlgoliaSearchHelper.prototype.toggleRefine = function() {
    return this.toggleFacetRefinement.apply(this, arguments);
};
/**
 * Adds or removes a tag filter with the `value` provided. If
 * the value is set then it removes it, otherwise it adds the filter.
 *
 * This method resets the current page to 0.
 * @param {string} tag tag to remove or add
 * @return {AlgoliaSearchHelper}
 * @fires change
 * @chainable
 */ AlgoliaSearchHelper.prototype.toggleTag = function(tag) {
    this._change({
        state: this.state.resetPage().toggleTagRefinement(tag),
        isPageReset: true
    });
    return this;
};
/**
 * Increments the page number by one.
 * @return {AlgoliaSearchHelper}
 * @fires change
 * @chainable
 * @example
 * helper.setPage(0).nextPage().getPage();
 * // returns 1
 */ AlgoliaSearchHelper.prototype.nextPage = function() {
    var page = this.state.page || 0;
    return this.setPage(page + 1);
};
/**
 * Decrements the page number by one.
 * @fires change
 * @return {AlgoliaSearchHelper}
 * @chainable
 * @example
 * helper.setPage(1).previousPage().getPage();
 * // returns 0
 */ AlgoliaSearchHelper.prototype.previousPage = function() {
    var page = this.state.page || 0;
    return this.setPage(page - 1);
};
/**
 * @private
 */ function setCurrentPage(page) {
    if (page < 0) throw new Error('Page requested below 0.');
    this._change({
        state: this.state.setPage(page),
        isPageReset: false
    });
    return this;
}
/**
 * Change the current page
 * @deprecated
 * @param  {number} page The page number
 * @return {AlgoliaSearchHelper}
 * @fires change
 * @chainable
 */ AlgoliaSearchHelper.prototype.setCurrentPage = setCurrentPage;
/**
 * Updates the current page.
 * @function
 * @param  {number} page The page number
 * @return {AlgoliaSearchHelper}
 * @fires change
 * @chainable
 */ AlgoliaSearchHelper.prototype.setPage = setCurrentPage;
/**
 * Updates the name of the index that will be targeted by the query.
 *
 * This method resets the current page to 0.
 * @param {string} name the index name
 * @return {AlgoliaSearchHelper}
 * @fires change
 * @chainable
 */ AlgoliaSearchHelper.prototype.setIndex = function(name) {
    this._change({
        state: this.state.resetPage().setIndex(name),
        isPageReset: true
    });
    return this;
};
/**
 * Update a parameter of the search. This method reset the page
 *
 * The complete list of parameters is available on the
 * [Algolia website](https://www.algolia.com/doc/rest#query-an-index).
 * The most commonly used parameters have their own [shortcuts](#query-parameters-shortcuts)
 * or benefit from higher-level APIs (all the kind of filters and facets have their own API)
 *
 * This method resets the current page to 0.
 * @param {string} parameter name of the parameter to update
 * @param {any} value new value of the parameter
 * @return {AlgoliaSearchHelper}
 * @fires change
 * @chainable
 * @example
 * helper.setQueryParameter('hitsPerPage', 20).search();
 */ AlgoliaSearchHelper.prototype.setQueryParameter = function(parameter, value) {
    this._change({
        state: this.state.resetPage().setQueryParameter(parameter, value),
        isPageReset: true
    });
    return this;
};
/**
 * Set the whole state (warning: will erase previous state)
 * @param {SearchParameters} newState the whole new state
 * @return {AlgoliaSearchHelper}
 * @fires change
 * @chainable
 */ AlgoliaSearchHelper.prototype.setState = function(newState) {
    this._change({
        state: SearchParameters.make(newState),
        isPageReset: false
    });
    return this;
};
/**
 * Override the current state without triggering a change event.
 * Do not use this method unless you know what you are doing. (see the example
 * for a legit use case)
 * @param {SearchParameters} newState the whole new state
 * @return {AlgoliaSearchHelper}
 * @example
 *  helper.on('change', function(state){
 *    // In this function you might want to find a way to store the state in the url/history
 *    updateYourURL(state)
 *  })
 *  window.onpopstate = function(event){
 *    // This is naive though as you should check if the state is really defined etc.
 *    helper.overrideStateWithoutTriggeringChangeEvent(event.state).search()
 *  }
 * @chainable
 */ AlgoliaSearchHelper.prototype.overrideStateWithoutTriggeringChangeEvent = function(newState) {
    this.state = new SearchParameters(newState);
    return this;
};
/**
 * Check if an attribute has any numeric, conjunctive, disjunctive or hierarchical filters.
 * @param {string} attribute the name of the attribute
 * @return {boolean} true if the attribute is filtered by at least one value
 * @example
 * // hasRefinements works with numeric, conjunctive, disjunctive and hierarchical filters
 * helper.hasRefinements('price'); // false
 * helper.addNumericRefinement('price', '>', 100);
 * helper.hasRefinements('price'); // true
 *
 * helper.hasRefinements('color'); // false
 * helper.addFacetRefinement('color', 'blue');
 * helper.hasRefinements('color'); // true
 *
 * helper.hasRefinements('material'); // false
 * helper.addDisjunctiveFacetRefinement('material', 'plastic');
 * helper.hasRefinements('material'); // true
 *
 * helper.hasRefinements('categories'); // false
 * helper.toggleFacetRefinement('categories', 'kitchen > knife');
 * helper.hasRefinements('categories'); // true
 *
 */ AlgoliaSearchHelper.prototype.hasRefinements = function(attribute) {
    if (objectHasKeys(this.state.getNumericRefinements(attribute))) return true;
    else if (this.state.isConjunctiveFacet(attribute)) return this.state.isFacetRefined(attribute);
    else if (this.state.isDisjunctiveFacet(attribute)) return this.state.isDisjunctiveFacetRefined(attribute);
    else if (this.state.isHierarchicalFacet(attribute)) return this.state.isHierarchicalFacetRefined(attribute);
    // there's currently no way to know that the user did call `addNumericRefinement` at some point
    // thus we cannot distinguish if there once was a numeric refinement that was cleared
    // so we will return false in every other situations to be consistent
    // while what we should do here is throw because we did not find the attribute in any type
    // of refinement
    return false;
};
/**
 * Check if a value is excluded for a specific faceted attribute. If the value
 * is omitted then the function checks if there is any excluding refinements.
 *
 * @param  {string}  facet name of the attribute for used for faceting
 * @param  {string}  [value] optional value. If passed will test that this value
   * is filtering the given facet.
 * @return {boolean} true if refined
 * @example
 * helper.isExcludeRefined('color'); // false
 * helper.isExcludeRefined('color', 'blue') // false
 * helper.isExcludeRefined('color', 'red') // false
 *
 * helper.addFacetExclusion('color', 'red');
 *
 * helper.isExcludeRefined('color'); // true
 * helper.isExcludeRefined('color', 'blue') // false
 * helper.isExcludeRefined('color', 'red') // true
 */ AlgoliaSearchHelper.prototype.isExcluded = function(facet, value) {
    return this.state.isExcludeRefined(facet, value);
};
/**
 * @deprecated since 2.4.0, see {@link AlgoliaSearchHelper#hasRefinements}
 */ AlgoliaSearchHelper.prototype.isDisjunctiveRefined = function(facet, value) {
    return this.state.isDisjunctiveFacetRefined(facet, value);
};
/**
 * Check if the string is a currently filtering tag.
 * @param {string} tag tag to check
 * @return {boolean}
 */ AlgoliaSearchHelper.prototype.hasTag = function(tag) {
    return this.state.isTagRefined(tag);
};
/**
 * @deprecated since 2.4.0, see {@link AlgoliaSearchHelper#hasTag}
 */ AlgoliaSearchHelper.prototype.isTagRefined = function() {
    return this.hasTagRefinements.apply(this, arguments);
};
/**
 * Get the name of the currently used index.
 * @return {string}
 * @example
 * helper.setIndex('highestPrice_products').getIndex();
 * // returns 'highestPrice_products'
 */ AlgoliaSearchHelper.prototype.getIndex = function() {
    return this.state.index;
};
function getCurrentPage() {
    return this.state.page;
}
/**
 * Get the currently selected page
 * @deprecated
 * @return {number} the current page
 */ AlgoliaSearchHelper.prototype.getCurrentPage = getCurrentPage;
/**
 * Get the currently selected page
 * @function
 * @return {number} the current page
 */ AlgoliaSearchHelper.prototype.getPage = getCurrentPage;
/**
 * Get all the tags currently set to filters the results.
 *
 * @return {string[]} The list of tags currently set.
 */ AlgoliaSearchHelper.prototype.getTags = function() {
    return this.state.tagRefinements;
};
/**
 * Get the list of refinements for a given attribute. This method works with
 * conjunctive, disjunctive, excluding and numerical filters.
 *
 * See also SearchResults#getRefinements
 *
 * @param {string} facetName attribute name used for faceting
 * @return {Array.<FacetRefinement|NumericRefinement>} All Refinement are objects that contain a value, and
 * a type. Numeric also contains an operator.
 * @example
 * helper.addNumericRefinement('price', '>', 100);
 * helper.getRefinements('price');
 * // [
 * //   {
 * //     "value": [
 * //       100
 * //     ],
 * //     "operator": ">",
 * //     "type": "numeric"
 * //   }
 * // ]
 * @example
 * helper.addFacetRefinement('color', 'blue');
 * helper.addFacetExclusion('color', 'red');
 * helper.getRefinements('color');
 * // [
 * //   {
 * //     "value": "blue",
 * //     "type": "conjunctive"
 * //   },
 * //   {
 * //     "value": "red",
 * //     "type": "exclude"
 * //   }
 * // ]
 * @example
 * helper.addDisjunctiveFacetRefinement('material', 'plastic');
 * // [
 * //   {
 * //     "value": "plastic",
 * //     "type": "disjunctive"
 * //   }
 * // ]
 */ AlgoliaSearchHelper.prototype.getRefinements = function(facetName) {
    var refinements = [];
    if (this.state.isConjunctiveFacet(facetName)) {
        var conjRefinements = this.state.getConjunctiveRefinements(facetName);
        conjRefinements.forEach(function(r) {
            refinements.push({
                value: r,
                type: 'conjunctive'
            });
        });
        var excludeRefinements = this.state.getExcludeRefinements(facetName);
        excludeRefinements.forEach(function(r) {
            refinements.push({
                value: r,
                type: 'exclude'
            });
        });
    } else if (this.state.isDisjunctiveFacet(facetName)) {
        var disjRefinements = this.state.getDisjunctiveRefinements(facetName);
        disjRefinements.forEach(function(r) {
            refinements.push({
                value: r,
                type: 'disjunctive'
            });
        });
    }
    var numericRefinements = this.state.getNumericRefinements(facetName);
    Object.keys(numericRefinements).forEach(function(operator) {
        var value = numericRefinements[operator];
        refinements.push({
            value: value,
            operator: operator,
            type: 'numeric'
        });
    });
    return refinements;
};
/**
 * Return the current refinement for the (attribute, operator)
 * @param {string} attribute attribute in the record
 * @param {string} operator operator applied on the refined values
 * @return {Array.<number|number[]>} refined values
 */ AlgoliaSearchHelper.prototype.getNumericRefinement = function(attribute, operator) {
    return this.state.getNumericRefinement(attribute, operator);
};
/**
 * Get the current breadcrumb for a hierarchical facet, as an array
 * @param  {string} facetName Hierarchical facet name
 * @return {array.<string>} the path as an array of string
 */ AlgoliaSearchHelper.prototype.getHierarchicalFacetBreadcrumb = function(facetName) {
    return this.state.getHierarchicalFacetBreadcrumb(facetName);
};
// /////////// PRIVATE
/**
 * Perform the underlying queries
 * @private
 * @return {undefined}
 * @fires search
 * @fires result
 * @fires error
 */ AlgoliaSearchHelper.prototype._search = function(options) {
    var state = this.state;
    var states = [];
    var mainQueries = [];
    if (!options.onlyWithDerivedHelpers) {
        mainQueries = requestBuilder._getQueries(state.index, state);
        states.push({
            state: state,
            queriesCount: mainQueries.length,
            helper: this
        });
        this.emit('search', {
            state: state,
            results: this.lastResults
        });
    }
    var derivedQueries = this.derivedHelpers.map(function(derivedHelper) {
        var derivedState = derivedHelper.getModifiedState(state);
        var derivedStateQueries = requestBuilder._getQueries(derivedState.index, derivedState);
        states.push({
            state: derivedState,
            queriesCount: derivedStateQueries.length,
            helper: derivedHelper
        });
        derivedHelper.emit('search', {
            state: derivedState,
            results: derivedHelper.lastResults
        });
        return derivedStateQueries;
    });
    var queries = Array.prototype.concat.apply(mainQueries, derivedQueries);
    var queryId = this._queryId++;
    this._currentNbQueries++;
    try {
        this.client.search(queries).then(this._dispatchAlgoliaResponse.bind(this, states, queryId)).catch(this._dispatchAlgoliaError.bind(this, queryId));
    } catch (error) {
        // If we reach this part, we're in an internal error state
        this.emit('error', {
            error: error
        });
    }
};
/**
 * Transform the responses as sent by the server and transform them into a user
 * usable object that merge the results of all the batch requests. It will dispatch
 * over the different helper + derived helpers (when there are some).
 * @private
 * @param {array.<{SearchParameters, AlgoliaQueries, AlgoliaSearchHelper}>}
 *  state state used for to generate the request
 * @param {number} queryId id of the current request
 * @param {object} content content of the response
 * @return {undefined}
 */ AlgoliaSearchHelper.prototype._dispatchAlgoliaResponse = function(states, queryId, content) {
    // FIXME remove the number of outdated queries discarded instead of just one
    if (queryId < this._lastQueryIdReceived) // Outdated answer
    return;
    this._currentNbQueries -= queryId - this._lastQueryIdReceived;
    this._lastQueryIdReceived = queryId;
    if (this._currentNbQueries === 0) this.emit('searchQueueEmpty');
    var results = content.results.slice();
    states.forEach(function(s) {
        var state = s.state;
        var queriesCount = s.queriesCount;
        var helper = s.helper;
        var specificResults = results.splice(0, queriesCount);
        var formattedResponse = helper.lastResults = new SearchResults(state, specificResults);
        helper.emit('result', {
            results: formattedResponse,
            state: state
        });
    });
};
AlgoliaSearchHelper.prototype._dispatchAlgoliaError = function(queryId, error) {
    if (queryId < this._lastQueryIdReceived) // Outdated answer
    return;
    this._currentNbQueries -= queryId - this._lastQueryIdReceived;
    this._lastQueryIdReceived = queryId;
    this.emit('error', {
        error: error
    });
    if (this._currentNbQueries === 0) this.emit('searchQueueEmpty');
};
AlgoliaSearchHelper.prototype.containsRefinement = function(query, facetFilters, numericFilters, tagFilters) {
    return query || facetFilters.length !== 0 || numericFilters.length !== 0 || tagFilters.length !== 0;
};
/**
 * Test if there are some disjunctive refinements on the facet
 * @private
 * @param {string} facet the attribute to test
 * @return {boolean}
 */ AlgoliaSearchHelper.prototype._hasDisjunctiveRefinements = function(facet) {
    return this.state.disjunctiveRefinements[facet] && this.state.disjunctiveRefinements[facet].length > 0;
};
AlgoliaSearchHelper.prototype._change = function(event) {
    var state = event.state;
    var isPageReset = event.isPageReset;
    if (state !== this.state) {
        this.state = state;
        this.emit('change', {
            state: this.state,
            results: this.lastResults,
            isPageReset: isPageReset
        });
    }
};
/**
 * Clears the cache of the underlying Algolia client.
 * @return {AlgoliaSearchHelper}
 */ AlgoliaSearchHelper.prototype.clearCache = function() {
    this.client.clearCache && this.client.clearCache();
    return this;
};
/**
 * Updates the internal client instance. If the reference of the clients
 * are equal then no update is actually done.
 * @param  {AlgoliaSearch} newClient an AlgoliaSearch client
 * @return {AlgoliaSearchHelper}
 */ AlgoliaSearchHelper.prototype.setClient = function(newClient) {
    if (this.client === newClient) return this;
    if (typeof newClient.addAlgoliaAgent === 'function') newClient.addAlgoliaAgent('JS Helper (' + version + ')');
    this.client = newClient;
    return this;
};
/**
 * Gets the instance of the currently used client.
 * @return {AlgoliaSearch}
 */ AlgoliaSearchHelper.prototype.getClient = function() {
    return this.client;
};
/**
 * Creates an derived instance of the Helper. A derived helper
 * is a way to request other indices synchronised with the lifecycle
 * of the main Helper. This mechanism uses the multiqueries feature
 * of Algolia to aggregate all the requests in a single network call.
 *
 * This method takes a function that is used to create a new SearchParameter
 * that will be used to create requests to Algolia. Those new requests
 * are created just before the `search` event. The signature of the function
 * is `SearchParameters -> SearchParameters`.
 *
 * This method returns a new DerivedHelper which is an EventEmitter
 * that fires the same `search`, `result` and `error` events. Those
 * events, however, will receive data specific to this DerivedHelper
 * and the SearchParameters that is returned by the call of the
 * parameter function.
 * @param {function} fn SearchParameters -> SearchParameters
 * @return {DerivedHelper}
 */ AlgoliaSearchHelper.prototype.derive = function(fn) {
    var derivedHelper = new DerivedHelper(this, fn);
    this.derivedHelpers.push(derivedHelper);
    return derivedHelper;
};
/**
 * This method detaches a derived Helper from the main one. Prefer using the one from the
 * derived helper itself, to remove the event listeners too.
 * @private
 * @return {undefined}
 * @throws Error
 */ AlgoliaSearchHelper.prototype.detachDerivedHelper = function(derivedHelper) {
    var pos = this.derivedHelpers.indexOf(derivedHelper);
    if (pos === -1) throw new Error('Derived helper already detached');
    this.derivedHelpers.splice(pos, 1);
};
/**
 * This method returns true if there is currently at least one on-going search.
 * @return {boolean} true if there is a search pending
 */ AlgoliaSearchHelper.prototype.hasPendingRequests = function() {
    return this._currentNbQueries > 0;
};
/**
 * @typedef AlgoliaSearchHelper.NumericRefinement
 * @type {object}
 * @property {number[]} value the numbers that are used for filtering this attribute with
 * the operator specified.
 * @property {string} operator the faceting data: value, number of entries
 * @property {string} type will be 'numeric'
 */ /**
 * @typedef AlgoliaSearchHelper.FacetRefinement
 * @type {object}
 * @property {string} value the string use to filter the attribute
 * @property {string} type the type of filter: 'conjunctive', 'disjunctive', 'exclude'
 */ module.exports = AlgoliaSearchHelper;

},{"./SearchParameters":"dQfwH","./SearchResults":"lUGU6","./DerivedHelper":"6UDS7","./requestBuilder":"6rfof","@algolia/events":"euNDO","./functions/inherits":"a0E30","./functions/objectHasKeys":"alqSr","./functions/omit":"l3IzD","./functions/merge":"eGyc5","./version":"cs17k","./functions/escapeFacetValue":"3r1Qc"}],"dQfwH":[function(require,module,exports) {
'use strict';
var merge = require('../functions/merge');
var defaultsPure = require('../functions/defaultsPure');
var intersection = require('../functions/intersection');
var find = require('../functions/find');
var valToNumber = require('../functions/valToNumber');
var omit = require('../functions/omit');
var objectHasKeys = require('../functions/objectHasKeys');
var isValidUserToken = require('../utils/isValidUserToken');
var RefinementList = require('./RefinementList');
/**
 * isEqual, but only for numeric refinement values, possible values:
 * - 5
 * - [5]
 * - [[5]]
 * - [[5,5],[4]]
 */ function isEqualNumericRefinement(a, b) {
    if (Array.isArray(a) && Array.isArray(b)) return a.length === b.length && a.every(function(el, i) {
        return isEqualNumericRefinement(b[i], el);
    });
    return a === b;
}
/**
 * like _.find but using deep equality to be able to use it
 * to find arrays.
 * @private
 * @param {any[]} array array to search into (elements are base or array of base)
 * @param {any} searchedValue the value we're looking for (base or array of base)
 * @return {any} the searched value or undefined
 */ function findArray(array, searchedValue) {
    return find(array, function(currentValue) {
        return isEqualNumericRefinement(currentValue, searchedValue);
    });
}
/**
 * The facet list is the structure used to store the list of values used to
 * filter a single attribute.
 * @typedef {string[]} SearchParameters.FacetList
 */ /**
 * Structure to store numeric filters with the operator as the key. The supported operators
 * are `=`, `>`, `<`, `>=`, `<=` and `!=`.
 * @typedef {Object.<string, Array.<number|number[]>>} SearchParameters.OperatorList
 */ /**
 * SearchParameters is the data structure that contains all the information
 * usable for making a search to Algolia API. It doesn't do the search itself,
 * nor does it contains logic about the parameters.
 * It is an immutable object, therefore it has been created in a way that each
 * changes does not change the object itself but returns a copy with the
 * modification.
 * This object should probably not be instantiated outside of the helper. It will
 * be provided when needed. This object is documented for reference as you'll
 * get it from events generated by the {@link AlgoliaSearchHelper}.
 * If need be, instantiate the Helper from the factory function {@link SearchParameters.make}
 * @constructor
 * @classdesc contains all the parameters of a search
 * @param {object|SearchParameters} newParameters existing parameters or partial object
 * for the properties of a new SearchParameters
 * @see SearchParameters.make
 * @example <caption>SearchParameters of the first query in
 *   <a href="http://demos.algolia.com/instant-search-demo/">the instant search demo</a></caption>
{
   "query": "",
   "disjunctiveFacets": [
      "customerReviewCount",
      "category",
      "salePrice_range",
      "manufacturer"
  ],
   "maxValuesPerFacet": 30,
   "page": 0,
   "hitsPerPage": 10,
   "facets": [
      "type",
      "shipping"
  ]
}
 */ function SearchParameters(newParameters) {
    var params = newParameters ? SearchParameters._parseNumbers(newParameters) : {};
    if (params.userToken !== undefined && !isValidUserToken(params.userToken)) console.warn('[algoliasearch-helper] The `userToken` parameter is invalid. This can lead to wrong analytics.\n  - Format: [a-zA-Z0-9_-]{1,64}');
    /**
   * This attribute contains the list of all the conjunctive facets
   * used. This list will be added to requested facets in the
   * [facets attribute](https://www.algolia.com/doc/rest-api/search#param-facets) sent to algolia.
   * @member {string[]}
   */ this.facets = params.facets || [];
    /**
   * This attribute contains the list of all the disjunctive facets
   * used. This list will be added to requested facets in the
   * [facets attribute](https://www.algolia.com/doc/rest-api/search#param-facets) sent to algolia.
   * @member {string[]}
   */ this.disjunctiveFacets = params.disjunctiveFacets || [];
    /**
   * This attribute contains the list of all the hierarchical facets
   * used. This list will be added to requested facets in the
   * [facets attribute](https://www.algolia.com/doc/rest-api/search#param-facets) sent to algolia.
   * Hierarchical facets are a sub type of disjunctive facets that
   * let you filter faceted attributes hierarchically.
   * @member {string[]|object[]}
   */ this.hierarchicalFacets = params.hierarchicalFacets || [];
    // Refinements
    /**
   * This attribute contains all the filters that need to be
   * applied on the conjunctive facets. Each facet must be properly
   * defined in the `facets` attribute.
   *
   * The key is the name of the facet, and the `FacetList` contains all
   * filters selected for the associated facet name.
   *
   * When querying algolia, the values stored in this attribute will
   * be translated into the `facetFilters` attribute.
   * @member {Object.<string, SearchParameters.FacetList>}
   */ this.facetsRefinements = params.facetsRefinements || {};
    /**
   * This attribute contains all the filters that need to be
   * excluded from the conjunctive facets. Each facet must be properly
   * defined in the `facets` attribute.
   *
   * The key is the name of the facet, and the `FacetList` contains all
   * filters excluded for the associated facet name.
   *
   * When querying algolia, the values stored in this attribute will
   * be translated into the `facetFilters` attribute.
   * @member {Object.<string, SearchParameters.FacetList>}
   */ this.facetsExcludes = params.facetsExcludes || {};
    /**
   * This attribute contains all the filters that need to be
   * applied on the disjunctive facets. Each facet must be properly
   * defined in the `disjunctiveFacets` attribute.
   *
   * The key is the name of the facet, and the `FacetList` contains all
   * filters selected for the associated facet name.
   *
   * When querying algolia, the values stored in this attribute will
   * be translated into the `facetFilters` attribute.
   * @member {Object.<string, SearchParameters.FacetList>}
   */ this.disjunctiveFacetsRefinements = params.disjunctiveFacetsRefinements || {};
    /**
   * This attribute contains all the filters that need to be
   * applied on the numeric attributes.
   *
   * The key is the name of the attribute, and the value is the
   * filters to apply to this attribute.
   *
   * When querying algolia, the values stored in this attribute will
   * be translated into the `numericFilters` attribute.
   * @member {Object.<string, SearchParameters.OperatorList>}
   */ this.numericRefinements = params.numericRefinements || {};
    /**
   * This attribute contains all the tags used to refine the query.
   *
   * When querying algolia, the values stored in this attribute will
   * be translated into the `tagFilters` attribute.
   * @member {string[]}
   */ this.tagRefinements = params.tagRefinements || [];
    /**
   * This attribute contains all the filters that need to be
   * applied on the hierarchical facets. Each facet must be properly
   * defined in the `hierarchicalFacets` attribute.
   *
   * The key is the name of the facet, and the `FacetList` contains all
   * filters selected for the associated facet name. The FacetList values
   * are structured as a string that contain the values for each level
   * separated by the configured separator.
   *
   * When querying algolia, the values stored in this attribute will
   * be translated into the `facetFilters` attribute.
   * @member {Object.<string, SearchParameters.FacetList>}
   */ this.hierarchicalFacetsRefinements = params.hierarchicalFacetsRefinements || {};
    var self = this;
    Object.keys(params).forEach(function(paramName) {
        var isKeyKnown = SearchParameters.PARAMETERS.indexOf(paramName) !== -1;
        var isValueDefined = params[paramName] !== undefined;
        if (!isKeyKnown && isValueDefined) self[paramName] = params[paramName];
    });
}
/**
 * List all the properties in SearchParameters and therefore all the known Algolia properties
 * This doesn't contain any beta/hidden features.
 * @private
 */ SearchParameters.PARAMETERS = Object.keys(new SearchParameters());
/**
 * @private
 * @param {object} partialState full or part of a state
 * @return {object} a new object with the number keys as number
 */ SearchParameters._parseNumbers = function(partialState) {
    // Do not reparse numbers in SearchParameters, they ought to be parsed already
    if (partialState instanceof SearchParameters) return partialState;
    var numbers = {};
    var numberKeys = [
        'aroundPrecision',
        'aroundRadius',
        'getRankingInfo',
        'minWordSizefor2Typos',
        'minWordSizefor1Typo',
        'page',
        'maxValuesPerFacet',
        'distinct',
        'minimumAroundRadius',
        'hitsPerPage',
        'minProximity'
    ];
    numberKeys.forEach(function(k) {
        var value = partialState[k];
        if (typeof value === 'string') {
            var parsedValue = parseFloat(value);
            // global isNaN is ok to use here, value is only number or NaN
            numbers[k] = isNaN(parsedValue) ? value : parsedValue;
        }
    });
    // there's two formats of insideBoundingBox, we need to parse
    // the one which is an array of float geo rectangles
    if (Array.isArray(partialState.insideBoundingBox)) numbers.insideBoundingBox = partialState.insideBoundingBox.map(function(geoRect) {
        if (Array.isArray(geoRect)) return geoRect.map(function(value) {
            return parseFloat(value);
        });
        return geoRect;
    });
    if (partialState.numericRefinements) {
        var numericRefinements = {};
        Object.keys(partialState.numericRefinements).forEach(function(attribute) {
            var operators = partialState.numericRefinements[attribute] || {};
            numericRefinements[attribute] = {};
            Object.keys(operators).forEach(function(operator) {
                var values = operators[operator];
                var parsedValues = values.map(function(v) {
                    if (Array.isArray(v)) return v.map(function(vPrime) {
                        if (typeof vPrime === 'string') return parseFloat(vPrime);
                        return vPrime;
                    });
                    else if (typeof v === 'string') return parseFloat(v);
                    return v;
                });
                numericRefinements[attribute][operator] = parsedValues;
            });
        });
        numbers.numericRefinements = numericRefinements;
    }
    return merge({}, partialState, numbers);
};
/**
 * Factory for SearchParameters
 * @param {object|SearchParameters} newParameters existing parameters or partial
 * object for the properties of a new SearchParameters
 * @return {SearchParameters} frozen instance of SearchParameters
 */ SearchParameters.make = function makeSearchParameters(newParameters) {
    var instance = new SearchParameters(newParameters);
    var hierarchicalFacets = newParameters.hierarchicalFacets || [];
    hierarchicalFacets.forEach(function(facet) {
        if (facet.rootPath) {
            var currentRefinement = instance.getHierarchicalRefinement(facet.name);
            if (currentRefinement.length > 0 && currentRefinement[0].indexOf(facet.rootPath) !== 0) instance = instance.clearRefinements(facet.name);
            // get it again in case it has been cleared
            currentRefinement = instance.getHierarchicalRefinement(facet.name);
            if (currentRefinement.length === 0) instance = instance.toggleHierarchicalFacetRefinement(facet.name, facet.rootPath);
        }
    });
    return instance;
};
/**
 * Validates the new parameters based on the previous state
 * @param {SearchParameters} currentState the current state
 * @param {object|SearchParameters} parameters the new parameters to set
 * @return {Error|null} Error if the modification is invalid, null otherwise
 */ SearchParameters.validate = function(currentState, parameters) {
    var params = parameters || {};
    if (currentState.tagFilters && params.tagRefinements && params.tagRefinements.length > 0) return new Error("[Tags] Cannot switch from the managed tag API to the advanced API. It is probably an error, if it is really what you want, you should first clear the tags with clearTags method.");
    if (currentState.tagRefinements.length > 0 && params.tagFilters) return new Error("[Tags] Cannot switch from the advanced tag API to the managed API. It is probably an error, if it is not, you should first clear the tags with clearTags method.");
    if (currentState.numericFilters && params.numericRefinements && objectHasKeys(params.numericRefinements)) return new Error("[Numeric filters] Can't switch from the advanced to the managed API. It is probably an error, if this is really what you want, you have to first clear the numeric filters.");
    if (objectHasKeys(currentState.numericRefinements) && params.numericFilters) return new Error("[Numeric filters] Can't switch from the managed API to the advanced. It is probably an error, if this is really what you want, you have to first clear the numeric filters.");
    return null;
};
SearchParameters.prototype = {
    constructor: SearchParameters,
    /**
   * Remove all refinements (disjunctive + conjunctive + excludes + numeric filters)
   * @method
   * @param {undefined|string|SearchParameters.clearCallback} [attribute] optional string or function
   * - If not given, means to clear all the filters.
   * - If `string`, means to clear all refinements for the `attribute` named filter.
   * - If `function`, means to clear all the refinements that return truthy values.
   * @return {SearchParameters}
   */ clearRefinements: function clearRefinements(attribute) {
        var patch = {
            numericRefinements: this._clearNumericRefinements(attribute),
            facetsRefinements: RefinementList.clearRefinement(this.facetsRefinements, attribute, 'conjunctiveFacet'),
            facetsExcludes: RefinementList.clearRefinement(this.facetsExcludes, attribute, 'exclude'),
            disjunctiveFacetsRefinements: RefinementList.clearRefinement(this.disjunctiveFacetsRefinements, attribute, 'disjunctiveFacet'),
            hierarchicalFacetsRefinements: RefinementList.clearRefinement(this.hierarchicalFacetsRefinements, attribute, 'hierarchicalFacet')
        };
        if (patch.numericRefinements === this.numericRefinements && patch.facetsRefinements === this.facetsRefinements && patch.facetsExcludes === this.facetsExcludes && patch.disjunctiveFacetsRefinements === this.disjunctiveFacetsRefinements && patch.hierarchicalFacetsRefinements === this.hierarchicalFacetsRefinements) return this;
        return this.setQueryParameters(patch);
    },
    /**
   * Remove all the refined tags from the SearchParameters
   * @method
   * @return {SearchParameters}
   */ clearTags: function clearTags() {
        if (this.tagFilters === undefined && this.tagRefinements.length === 0) return this;
        return this.setQueryParameters({
            tagFilters: undefined,
            tagRefinements: []
        });
    },
    /**
   * Set the index.
   * @method
   * @param {string} index the index name
   * @return {SearchParameters}
   */ setIndex: function setIndex(index) {
        if (index === this.index) return this;
        return this.setQueryParameters({
            index: index
        });
    },
    /**
   * Query setter
   * @method
   * @param {string} newQuery value for the new query
   * @return {SearchParameters}
   */ setQuery: function setQuery(newQuery) {
        if (newQuery === this.query) return this;
        return this.setQueryParameters({
            query: newQuery
        });
    },
    /**
   * Page setter
   * @method
   * @param {number} newPage new page number
   * @return {SearchParameters}
   */ setPage: function setPage(newPage) {
        if (newPage === this.page) return this;
        return this.setQueryParameters({
            page: newPage
        });
    },
    /**
   * Facets setter
   * The facets are the simple facets, used for conjunctive (and) faceting.
   * @method
   * @param {string[]} facets all the attributes of the algolia records used for conjunctive faceting
   * @return {SearchParameters}
   */ setFacets: function setFacets(facets) {
        return this.setQueryParameters({
            facets: facets
        });
    },
    /**
   * Disjunctive facets setter
   * Change the list of disjunctive (or) facets the helper chan handle.
   * @method
   * @param {string[]} facets all the attributes of the algolia records used for disjunctive faceting
   * @return {SearchParameters}
   */ setDisjunctiveFacets: function setDisjunctiveFacets(facets) {
        return this.setQueryParameters({
            disjunctiveFacets: facets
        });
    },
    /**
   * HitsPerPage setter
   * Hits per page represents the number of hits retrieved for this query
   * @method
   * @param {number} n number of hits retrieved per page of results
   * @return {SearchParameters}
   */ setHitsPerPage: function setHitsPerPage(n) {
        if (this.hitsPerPage === n) return this;
        return this.setQueryParameters({
            hitsPerPage: n
        });
    },
    /**
   * typoTolerance setter
   * Set the value of typoTolerance
   * @method
   * @param {string} typoTolerance new value of typoTolerance ("true", "false", "min" or "strict")
   * @return {SearchParameters}
   */ setTypoTolerance: function setTypoTolerance(typoTolerance) {
        if (this.typoTolerance === typoTolerance) return this;
        return this.setQueryParameters({
            typoTolerance: typoTolerance
        });
    },
    /**
   * Add a numeric filter for a given attribute
   * When value is an array, they are combined with OR
   * When value is a single value, it will combined with AND
   * @method
   * @param {string} attribute attribute to set the filter on
   * @param {string} operator operator of the filter (possible values: =, >, >=, <, <=, !=)
   * @param {number | number[]} value value of the filter
   * @return {SearchParameters}
   * @example
   * // for price = 50 or 40
   * searchparameter.addNumericRefinement('price', '=', [50, 40]);
   * @example
   * // for size = 38 and 40
   * searchparameter.addNumericRefinement('size', '=', 38);
   * searchparameter.addNumericRefinement('size', '=', 40);
   */ addNumericRefinement: function(attribute, operator, v) {
        var value = valToNumber(v);
        if (this.isNumericRefined(attribute, operator, value)) return this;
        var mod = merge({}, this.numericRefinements);
        mod[attribute] = merge({}, mod[attribute]);
        if (mod[attribute][operator]) {
            // Array copy
            mod[attribute][operator] = mod[attribute][operator].slice();
            // Add the element. Concat can't be used here because value can be an array.
            mod[attribute][operator].push(value);
        } else mod[attribute][operator] = [
            value
        ];
        return this.setQueryParameters({
            numericRefinements: mod
        });
    },
    /**
   * Get the list of conjunctive refinements for a single facet
   * @param {string} facetName name of the attribute used for faceting
   * @return {string[]} list of refinements
   */ getConjunctiveRefinements: function(facetName) {
        if (!this.isConjunctiveFacet(facetName)) return [];
        return this.facetsRefinements[facetName] || [];
    },
    /**
   * Get the list of disjunctive refinements for a single facet
   * @param {string} facetName name of the attribute used for faceting
   * @return {string[]} list of refinements
   */ getDisjunctiveRefinements: function(facetName) {
        if (!this.isDisjunctiveFacet(facetName)) return [];
        return this.disjunctiveFacetsRefinements[facetName] || [];
    },
    /**
   * Get the list of hierarchical refinements for a single facet
   * @param {string} facetName name of the attribute used for faceting
   * @return {string[]} list of refinements
   */ getHierarchicalRefinement: function(facetName) {
        // we send an array but we currently do not support multiple
        // hierarchicalRefinements for a hierarchicalFacet
        return this.hierarchicalFacetsRefinements[facetName] || [];
    },
    /**
   * Get the list of exclude refinements for a single facet
   * @param {string} facetName name of the attribute used for faceting
   * @return {string[]} list of refinements
   */ getExcludeRefinements: function(facetName) {
        if (!this.isConjunctiveFacet(facetName)) return [];
        return this.facetsExcludes[facetName] || [];
    },
    /**
   * Remove all the numeric filter for a given (attribute, operator)
   * @method
   * @param {string} attribute attribute to set the filter on
   * @param {string} [operator] operator of the filter (possible values: =, >, >=, <, <=, !=)
   * @param {number} [number] the value to be removed
   * @return {SearchParameters}
   */ removeNumericRefinement: function(attribute, operator, paramValue) {
        if (paramValue !== undefined) {
            if (!this.isNumericRefined(attribute, operator, paramValue)) return this;
            return this.setQueryParameters({
                numericRefinements: this._clearNumericRefinements(function(value, key) {
                    return key === attribute && value.op === operator && isEqualNumericRefinement(value.val, valToNumber(paramValue));
                })
            });
        } else if (operator !== undefined) {
            if (!this.isNumericRefined(attribute, operator)) return this;
            return this.setQueryParameters({
                numericRefinements: this._clearNumericRefinements(function(value, key) {
                    return key === attribute && value.op === operator;
                })
            });
        }
        if (!this.isNumericRefined(attribute)) return this;
        return this.setQueryParameters({
            numericRefinements: this._clearNumericRefinements(function(value, key) {
                return key === attribute;
            })
        });
    },
    /**
   * Get the list of numeric refinements for a single facet
   * @param {string} facetName name of the attribute used for faceting
   * @return {SearchParameters.OperatorList} list of refinements
   */ getNumericRefinements: function(facetName) {
        return this.numericRefinements[facetName] || {};
    },
    /**
   * Return the current refinement for the (attribute, operator)
   * @param {string} attribute attribute in the record
   * @param {string} operator operator applied on the refined values
   * @return {Array.<number|number[]>} refined values
   */ getNumericRefinement: function(attribute, operator) {
        return this.numericRefinements[attribute] && this.numericRefinements[attribute][operator];
    },
    /**
   * Clear numeric filters.
   * @method
   * @private
   * @param {string|SearchParameters.clearCallback} [attribute] optional string or function
   * - If not given, means to clear all the filters.
   * - If `string`, means to clear all refinements for the `attribute` named filter.
   * - If `function`, means to clear all the refinements that return truthy values.
   * @return {Object.<string, OperatorList>}
   */ _clearNumericRefinements: function _clearNumericRefinements(attribute) {
        if (attribute === undefined) {
            if (!objectHasKeys(this.numericRefinements)) return this.numericRefinements;
            return {};
        } else if (typeof attribute === 'string') return omit(this.numericRefinements, [
            attribute
        ]);
        else if (typeof attribute === 'function') {
            var hasChanged = false;
            var numericRefinements = this.numericRefinements;
            var newNumericRefinements = Object.keys(numericRefinements).reduce(function(memo, key) {
                var operators = numericRefinements[key];
                var operatorList = {};
                operators = operators || {};
                Object.keys(operators).forEach(function(operator) {
                    var values = operators[operator] || [];
                    var outValues = [];
                    values.forEach(function(value) {
                        var predicateResult = attribute({
                            val: value,
                            op: operator
                        }, key, 'numeric');
                        if (!predicateResult) outValues.push(value);
                    });
                    if (outValues.length !== values.length) hasChanged = true;
                    operatorList[operator] = outValues;
                });
                memo[key] = operatorList;
                return memo;
            }, {});
            if (hasChanged) return newNumericRefinements;
            return this.numericRefinements;
        }
    },
    /**
   * Add a facet to the facets attribute of the helper configuration, if it
   * isn't already present.
   * @method
   * @param {string} facet facet name to add
   * @return {SearchParameters}
   */ addFacet: function addFacet(facet) {
        if (this.isConjunctiveFacet(facet)) return this;
        return this.setQueryParameters({
            facets: this.facets.concat([
                facet
            ])
        });
    },
    /**
   * Add a disjunctive facet to the disjunctiveFacets attribute of the helper
   * configuration, if it isn't already present.
   * @method
   * @param {string} facet disjunctive facet name to add
   * @return {SearchParameters}
   */ addDisjunctiveFacet: function addDisjunctiveFacet(facet) {
        if (this.isDisjunctiveFacet(facet)) return this;
        return this.setQueryParameters({
            disjunctiveFacets: this.disjunctiveFacets.concat([
                facet
            ])
        });
    },
    /**
   * Add a hierarchical facet to the hierarchicalFacets attribute of the helper
   * configuration.
   * @method
   * @param {object} hierarchicalFacet hierarchical facet to add
   * @return {SearchParameters}
   * @throws will throw an error if a hierarchical facet with the same name was already declared
   */ addHierarchicalFacet: function addHierarchicalFacet(hierarchicalFacet) {
        if (this.isHierarchicalFacet(hierarchicalFacet.name)) throw new Error('Cannot declare two hierarchical facets with the same name: `' + hierarchicalFacet.name + '`');
        return this.setQueryParameters({
            hierarchicalFacets: this.hierarchicalFacets.concat([
                hierarchicalFacet
            ])
        });
    },
    /**
   * Add a refinement on a "normal" facet
   * @method
   * @param {string} facet attribute to apply the faceting on
   * @param {string} value value of the attribute (will be converted to string)
   * @return {SearchParameters}
   */ addFacetRefinement: function addFacetRefinement(facet, value) {
        if (!this.isConjunctiveFacet(facet)) throw new Error(facet + ' is not defined in the facets attribute of the helper configuration');
        if (RefinementList.isRefined(this.facetsRefinements, facet, value)) return this;
        return this.setQueryParameters({
            facetsRefinements: RefinementList.addRefinement(this.facetsRefinements, facet, value)
        });
    },
    /**
   * Exclude a value from a "normal" facet
   * @method
   * @param {string} facet attribute to apply the exclusion on
   * @param {string} value value of the attribute (will be converted to string)
   * @return {SearchParameters}
   */ addExcludeRefinement: function addExcludeRefinement(facet, value) {
        if (!this.isConjunctiveFacet(facet)) throw new Error(facet + ' is not defined in the facets attribute of the helper configuration');
        if (RefinementList.isRefined(this.facetsExcludes, facet, value)) return this;
        return this.setQueryParameters({
            facetsExcludes: RefinementList.addRefinement(this.facetsExcludes, facet, value)
        });
    },
    /**
   * Adds a refinement on a disjunctive facet.
   * @method
   * @param {string} facet attribute to apply the faceting on
   * @param {string} value value of the attribute (will be converted to string)
   * @return {SearchParameters}
   */ addDisjunctiveFacetRefinement: function addDisjunctiveFacetRefinement(facet, value) {
        if (!this.isDisjunctiveFacet(facet)) throw new Error(facet + ' is not defined in the disjunctiveFacets attribute of the helper configuration');
        if (RefinementList.isRefined(this.disjunctiveFacetsRefinements, facet, value)) return this;
        return this.setQueryParameters({
            disjunctiveFacetsRefinements: RefinementList.addRefinement(this.disjunctiveFacetsRefinements, facet, value)
        });
    },
    /**
   * addTagRefinement adds a tag to the list used to filter the results
   * @param {string} tag tag to be added
   * @return {SearchParameters}
   */ addTagRefinement: function addTagRefinement(tag) {
        if (this.isTagRefined(tag)) return this;
        var modification = {
            tagRefinements: this.tagRefinements.concat(tag)
        };
        return this.setQueryParameters(modification);
    },
    /**
   * Remove a facet from the facets attribute of the helper configuration, if it
   * is present.
   * @method
   * @param {string} facet facet name to remove
   * @return {SearchParameters}
   */ removeFacet: function removeFacet(facet) {
        if (!this.isConjunctiveFacet(facet)) return this;
        return this.clearRefinements(facet).setQueryParameters({
            facets: this.facets.filter(function(f) {
                return f !== facet;
            })
        });
    },
    /**
   * Remove a disjunctive facet from the disjunctiveFacets attribute of the
   * helper configuration, if it is present.
   * @method
   * @param {string} facet disjunctive facet name to remove
   * @return {SearchParameters}
   */ removeDisjunctiveFacet: function removeDisjunctiveFacet(facet) {
        if (!this.isDisjunctiveFacet(facet)) return this;
        return this.clearRefinements(facet).setQueryParameters({
            disjunctiveFacets: this.disjunctiveFacets.filter(function(f) {
                return f !== facet;
            })
        });
    },
    /**
   * Remove a hierarchical facet from the hierarchicalFacets attribute of the
   * helper configuration, if it is present.
   * @method
   * @param {string} facet hierarchical facet name to remove
   * @return {SearchParameters}
   */ removeHierarchicalFacet: function removeHierarchicalFacet(facet) {
        if (!this.isHierarchicalFacet(facet)) return this;
        return this.clearRefinements(facet).setQueryParameters({
            hierarchicalFacets: this.hierarchicalFacets.filter(function(f) {
                return f.name !== facet;
            })
        });
    },
    /**
   * Remove a refinement set on facet. If a value is provided, it will clear the
   * refinement for the given value, otherwise it will clear all the refinement
   * values for the faceted attribute.
   * @method
   * @param {string} facet name of the attribute used for faceting
   * @param {string} [value] value used to filter
   * @return {SearchParameters}
   */ removeFacetRefinement: function removeFacetRefinement(facet, value) {
        if (!this.isConjunctiveFacet(facet)) throw new Error(facet + ' is not defined in the facets attribute of the helper configuration');
        if (!RefinementList.isRefined(this.facetsRefinements, facet, value)) return this;
        return this.setQueryParameters({
            facetsRefinements: RefinementList.removeRefinement(this.facetsRefinements, facet, value)
        });
    },
    /**
   * Remove a negative refinement on a facet
   * @method
   * @param {string} facet name of the attribute used for faceting
   * @param {string} value value used to filter
   * @return {SearchParameters}
   */ removeExcludeRefinement: function removeExcludeRefinement(facet, value) {
        if (!this.isConjunctiveFacet(facet)) throw new Error(facet + ' is not defined in the facets attribute of the helper configuration');
        if (!RefinementList.isRefined(this.facetsExcludes, facet, value)) return this;
        return this.setQueryParameters({
            facetsExcludes: RefinementList.removeRefinement(this.facetsExcludes, facet, value)
        });
    },
    /**
   * Remove a refinement on a disjunctive facet
   * @method
   * @param {string} facet name of the attribute used for faceting
   * @param {string} value value used to filter
   * @return {SearchParameters}
   */ removeDisjunctiveFacetRefinement: function removeDisjunctiveFacetRefinement(facet, value) {
        if (!this.isDisjunctiveFacet(facet)) throw new Error(facet + ' is not defined in the disjunctiveFacets attribute of the helper configuration');
        if (!RefinementList.isRefined(this.disjunctiveFacetsRefinements, facet, value)) return this;
        return this.setQueryParameters({
            disjunctiveFacetsRefinements: RefinementList.removeRefinement(this.disjunctiveFacetsRefinements, facet, value)
        });
    },
    /**
   * Remove a tag from the list of tag refinements
   * @method
   * @param {string} tag the tag to remove
   * @return {SearchParameters}
   */ removeTagRefinement: function removeTagRefinement(tag) {
        if (!this.isTagRefined(tag)) return this;
        var modification = {
            tagRefinements: this.tagRefinements.filter(function(t) {
                return t !== tag;
            })
        };
        return this.setQueryParameters(modification);
    },
    /**
   * Generic toggle refinement method to use with facet, disjunctive facets
   * and hierarchical facets
   * @param  {string} facet the facet to refine
   * @param  {string} value the associated value
   * @return {SearchParameters}
   * @throws will throw an error if the facet is not declared in the settings of the helper
   * @deprecated since version 2.19.0, see {@link SearchParameters#toggleFacetRefinement}
   */ toggleRefinement: function toggleRefinement(facet, value) {
        return this.toggleFacetRefinement(facet, value);
    },
    /**
   * Generic toggle refinement method to use with facet, disjunctive facets
   * and hierarchical facets
   * @param  {string} facet the facet to refine
   * @param  {string} value the associated value
   * @return {SearchParameters}
   * @throws will throw an error if the facet is not declared in the settings of the helper
   */ toggleFacetRefinement: function toggleFacetRefinement(facet, value) {
        if (this.isHierarchicalFacet(facet)) return this.toggleHierarchicalFacetRefinement(facet, value);
        else if (this.isConjunctiveFacet(facet)) return this.toggleConjunctiveFacetRefinement(facet, value);
        else if (this.isDisjunctiveFacet(facet)) return this.toggleDisjunctiveFacetRefinement(facet, value);
        throw new Error('Cannot refine the undeclared facet ' + facet + '; it should be added to the helper options facets, disjunctiveFacets or hierarchicalFacets');
    },
    /**
   * Switch the refinement applied over a facet/value
   * @method
   * @param {string} facet name of the attribute used for faceting
   * @param {value} value value used for filtering
   * @return {SearchParameters}
   */ toggleConjunctiveFacetRefinement: function toggleConjunctiveFacetRefinement(facet, value) {
        if (!this.isConjunctiveFacet(facet)) throw new Error(facet + ' is not defined in the facets attribute of the helper configuration');
        return this.setQueryParameters({
            facetsRefinements: RefinementList.toggleRefinement(this.facetsRefinements, facet, value)
        });
    },
    /**
   * Switch the refinement applied over a facet/value
   * @method
   * @param {string} facet name of the attribute used for faceting
   * @param {value} value value used for filtering
   * @return {SearchParameters}
   */ toggleExcludeFacetRefinement: function toggleExcludeFacetRefinement(facet, value) {
        if (!this.isConjunctiveFacet(facet)) throw new Error(facet + ' is not defined in the facets attribute of the helper configuration');
        return this.setQueryParameters({
            facetsExcludes: RefinementList.toggleRefinement(this.facetsExcludes, facet, value)
        });
    },
    /**
   * Switch the refinement applied over a facet/value
   * @method
   * @param {string} facet name of the attribute used for faceting
   * @param {value} value value used for filtering
   * @return {SearchParameters}
   */ toggleDisjunctiveFacetRefinement: function toggleDisjunctiveFacetRefinement(facet, value) {
        if (!this.isDisjunctiveFacet(facet)) throw new Error(facet + ' is not defined in the disjunctiveFacets attribute of the helper configuration');
        return this.setQueryParameters({
            disjunctiveFacetsRefinements: RefinementList.toggleRefinement(this.disjunctiveFacetsRefinements, facet, value)
        });
    },
    /**
   * Switch the refinement applied over a facet/value
   * @method
   * @param {string} facet name of the attribute used for faceting
   * @param {value} value value used for filtering
   * @return {SearchParameters}
   */ toggleHierarchicalFacetRefinement: function toggleHierarchicalFacetRefinement(facet, value) {
        if (!this.isHierarchicalFacet(facet)) throw new Error(facet + ' is not defined in the hierarchicalFacets attribute of the helper configuration');
        var separator = this._getHierarchicalFacetSeparator(this.getHierarchicalFacetByName(facet));
        var mod = {};
        var upOneOrMultipleLevel = this.hierarchicalFacetsRefinements[facet] !== undefined && this.hierarchicalFacetsRefinements[facet].length > 0 && // remove current refinement:
        // refinement was 'beer > IPA', call is toggleRefine('beer > IPA'), refinement should be `beer`
        (this.hierarchicalFacetsRefinements[facet][0] === value || // remove a parent refinement of the current refinement:
        //  - refinement was 'beer > IPA > Flying dog'
        //  - call is toggleRefine('beer > IPA')
        //  - refinement should be `beer`
        this.hierarchicalFacetsRefinements[facet][0].indexOf(value + separator) === 0);
        if (upOneOrMultipleLevel) {
            if (value.indexOf(separator) === -1) // go back to root level
            mod[facet] = [];
            else mod[facet] = [
                value.slice(0, value.lastIndexOf(separator))
            ];
        } else mod[facet] = [
            value
        ];
        return this.setQueryParameters({
            hierarchicalFacetsRefinements: defaultsPure({}, mod, this.hierarchicalFacetsRefinements)
        });
    },
    /**
   * Adds a refinement on a hierarchical facet.
   * @param {string} facet the facet name
   * @param {string} path the hierarchical facet path
   * @return {SearchParameter} the new state
   * @throws Error if the facet is not defined or if the facet is refined
   */ addHierarchicalFacetRefinement: function(facet, path) {
        if (this.isHierarchicalFacetRefined(facet)) throw new Error(facet + ' is already refined.');
        if (!this.isHierarchicalFacet(facet)) throw new Error(facet + ' is not defined in the hierarchicalFacets attribute of the helper configuration.');
        var mod = {};
        mod[facet] = [
            path
        ];
        return this.setQueryParameters({
            hierarchicalFacetsRefinements: defaultsPure({}, mod, this.hierarchicalFacetsRefinements)
        });
    },
    /**
   * Removes the refinement set on a hierarchical facet.
   * @param {string} facet the facet name
   * @return {SearchParameter} the new state
   * @throws Error if the facet is not defined or if the facet is not refined
   */ removeHierarchicalFacetRefinement: function(facet) {
        if (!this.isHierarchicalFacetRefined(facet)) return this;
        var mod = {};
        mod[facet] = [];
        return this.setQueryParameters({
            hierarchicalFacetsRefinements: defaultsPure({}, mod, this.hierarchicalFacetsRefinements)
        });
    },
    /**
   * Switch the tag refinement
   * @method
   * @param {string} tag the tag to remove or add
   * @return {SearchParameters}
   */ toggleTagRefinement: function toggleTagRefinement(tag) {
        if (this.isTagRefined(tag)) return this.removeTagRefinement(tag);
        return this.addTagRefinement(tag);
    },
    /**
   * Test if the facet name is from one of the disjunctive facets
   * @method
   * @param {string} facet facet name to test
   * @return {boolean}
   */ isDisjunctiveFacet: function(facet) {
        return this.disjunctiveFacets.indexOf(facet) > -1;
    },
    /**
   * Test if the facet name is from one of the hierarchical facets
   * @method
   * @param {string} facetName facet name to test
   * @return {boolean}
   */ isHierarchicalFacet: function(facetName) {
        return this.getHierarchicalFacetByName(facetName) !== undefined;
    },
    /**
   * Test if the facet name is from one of the conjunctive/normal facets
   * @method
   * @param {string} facet facet name to test
   * @return {boolean}
   */ isConjunctiveFacet: function(facet) {
        return this.facets.indexOf(facet) > -1;
    },
    /**
   * Returns true if the facet is refined, either for a specific value or in
   * general.
   * @method
   * @param {string} facet name of the attribute for used for faceting
   * @param {string} value, optional value. If passed will test that this value
   * is filtering the given facet.
   * @return {boolean} returns true if refined
   */ isFacetRefined: function isFacetRefined(facet, value) {
        if (!this.isConjunctiveFacet(facet)) return false;
        return RefinementList.isRefined(this.facetsRefinements, facet, value);
    },
    /**
   * Returns true if the facet contains exclusions or if a specific value is
   * excluded.
   *
   * @method
   * @param {string} facet name of the attribute for used for faceting
   * @param {string} [value] optional value. If passed will test that this value
   * is filtering the given facet.
   * @return {boolean} returns true if refined
   */ isExcludeRefined: function isExcludeRefined(facet, value) {
        if (!this.isConjunctiveFacet(facet)) return false;
        return RefinementList.isRefined(this.facetsExcludes, facet, value);
    },
    /**
   * Returns true if the facet contains a refinement, or if a value passed is a
   * refinement for the facet.
   * @method
   * @param {string} facet name of the attribute for used for faceting
   * @param {string} value optional, will test if the value is used for refinement
   * if there is one, otherwise will test if the facet contains any refinement
   * @return {boolean}
   */ isDisjunctiveFacetRefined: function isDisjunctiveFacetRefined(facet, value) {
        if (!this.isDisjunctiveFacet(facet)) return false;
        return RefinementList.isRefined(this.disjunctiveFacetsRefinements, facet, value);
    },
    /**
   * Returns true if the facet contains a refinement, or if a value passed is a
   * refinement for the facet.
   * @method
   * @param {string} facet name of the attribute for used for faceting
   * @param {string} value optional, will test if the value is used for refinement
   * if there is one, otherwise will test if the facet contains any refinement
   * @return {boolean}
   */ isHierarchicalFacetRefined: function isHierarchicalFacetRefined(facet, value) {
        if (!this.isHierarchicalFacet(facet)) return false;
        var refinements = this.getHierarchicalRefinement(facet);
        if (!value) return refinements.length > 0;
        return refinements.indexOf(value) !== -1;
    },
    /**
   * Test if the triple (attribute, operator, value) is already refined.
   * If only the attribute and the operator are provided, it tests if the
   * contains any refinement value.
   * @method
   * @param {string} attribute attribute for which the refinement is applied
   * @param {string} [operator] operator of the refinement
   * @param {string} [value] value of the refinement
   * @return {boolean} true if it is refined
   */ isNumericRefined: function isNumericRefined(attribute, operator, value) {
        if (value === undefined && operator === undefined) return !!this.numericRefinements[attribute];
        var isOperatorDefined = this.numericRefinements[attribute] && this.numericRefinements[attribute][operator] !== undefined;
        if (value === undefined || !isOperatorDefined) return isOperatorDefined;
        var parsedValue = valToNumber(value);
        var isAttributeValueDefined = findArray(this.numericRefinements[attribute][operator], parsedValue) !== undefined;
        return isOperatorDefined && isAttributeValueDefined;
    },
    /**
   * Returns true if the tag refined, false otherwise
   * @method
   * @param {string} tag the tag to check
   * @return {boolean}
   */ isTagRefined: function isTagRefined(tag) {
        return this.tagRefinements.indexOf(tag) !== -1;
    },
    /**
   * Returns the list of all disjunctive facets refined
   * @method
   * @param {string} facet name of the attribute used for faceting
   * @param {value} value value used for filtering
   * @return {string[]}
   */ getRefinedDisjunctiveFacets: function getRefinedDisjunctiveFacets() {
        var self = this;
        // attributes used for numeric filter can also be disjunctive
        var disjunctiveNumericRefinedFacets = intersection(Object.keys(this.numericRefinements).filter(function(facet) {
            return Object.keys(self.numericRefinements[facet]).length > 0;
        }), this.disjunctiveFacets);
        return Object.keys(this.disjunctiveFacetsRefinements).filter(function(facet) {
            return self.disjunctiveFacetsRefinements[facet].length > 0;
        }).concat(disjunctiveNumericRefinedFacets).concat(this.getRefinedHierarchicalFacets());
    },
    /**
   * Returns the list of all disjunctive facets refined
   * @method
   * @param {string} facet name of the attribute used for faceting
   * @param {value} value value used for filtering
   * @return {string[]}
   */ getRefinedHierarchicalFacets: function getRefinedHierarchicalFacets() {
        var self = this;
        return intersection(// enforce the order between the two arrays,
        // so that refinement name index === hierarchical facet index
        this.hierarchicalFacets.map(function(facet) {
            return facet.name;
        }), Object.keys(this.hierarchicalFacetsRefinements).filter(function(facet) {
            return self.hierarchicalFacetsRefinements[facet].length > 0;
        }));
    },
    /**
   * Returned the list of all disjunctive facets not refined
   * @method
   * @return {string[]}
   */ getUnrefinedDisjunctiveFacets: function() {
        var refinedFacets = this.getRefinedDisjunctiveFacets();
        return this.disjunctiveFacets.filter(function(f) {
            return refinedFacets.indexOf(f) === -1;
        });
    },
    managedParameters: [
        'index',
        'facets',
        'disjunctiveFacets',
        'facetsRefinements',
        'hierarchicalFacets',
        'facetsExcludes',
        'disjunctiveFacetsRefinements',
        'numericRefinements',
        'tagRefinements',
        'hierarchicalFacetsRefinements'
    ],
    getQueryParams: function getQueryParams() {
        var managedParameters = this.managedParameters;
        var queryParams = {};
        var self = this;
        Object.keys(this).forEach(function(paramName) {
            var paramValue = self[paramName];
            if (managedParameters.indexOf(paramName) === -1 && paramValue !== undefined) queryParams[paramName] = paramValue;
        });
        return queryParams;
    },
    /**
   * Let the user set a specific value for a given parameter. Will return the
   * same instance if the parameter is invalid or if the value is the same as the
   * previous one.
   * @method
   * @param {string} parameter the parameter name
   * @param {any} value the value to be set, must be compliant with the definition
   * of the attribute on the object
   * @return {SearchParameters} the updated state
   */ setQueryParameter: function setParameter(parameter, value) {
        if (this[parameter] === value) return this;
        var modification = {};
        modification[parameter] = value;
        return this.setQueryParameters(modification);
    },
    /**
   * Let the user set any of the parameters with a plain object.
   * @method
   * @param {object} params all the keys and the values to be updated
   * @return {SearchParameters} a new updated instance
   */ setQueryParameters: function setQueryParameters(params) {
        if (!params) return this;
        var error = SearchParameters.validate(this, params);
        if (error) throw error;
        var self = this;
        var nextWithNumbers = SearchParameters._parseNumbers(params);
        var previousPlainObject = Object.keys(this).reduce(function(acc, key) {
            acc[key] = self[key];
            return acc;
        }, {});
        var nextPlainObject = Object.keys(nextWithNumbers).reduce(function(previous, key) {
            var isPreviousValueDefined = previous[key] !== undefined;
            var isNextValueDefined = nextWithNumbers[key] !== undefined;
            if (isPreviousValueDefined && !isNextValueDefined) return omit(previous, [
                key
            ]);
            if (isNextValueDefined) previous[key] = nextWithNumbers[key];
            return previous;
        }, previousPlainObject);
        return new this.constructor(nextPlainObject);
    },
    /**
   * Returns a new instance with the page reset. Two scenarios possible:
   * the page is omitted -> return the given instance
   * the page is set -> return a new instance with a page of 0
   * @return {SearchParameters} a new updated instance
   */ resetPage: function() {
        if (this.page === undefined) return this;
        return this.setPage(0);
    },
    /**
   * Helper function to get the hierarchicalFacet separator or the default one (`>`)
   * @param  {object} hierarchicalFacet
   * @return {string} returns the hierarchicalFacet.separator or `>` as default
   */ _getHierarchicalFacetSortBy: function(hierarchicalFacet) {
        return hierarchicalFacet.sortBy || [
            'isRefined:desc',
            'name:asc'
        ];
    },
    /**
   * Helper function to get the hierarchicalFacet separator or the default one (`>`)
   * @private
   * @param  {object} hierarchicalFacet
   * @return {string} returns the hierarchicalFacet.separator or `>` as default
   */ _getHierarchicalFacetSeparator: function(hierarchicalFacet) {
        return hierarchicalFacet.separator || ' > ';
    },
    /**
   * Helper function to get the hierarchicalFacet prefix path or null
   * @private
   * @param  {object} hierarchicalFacet
   * @return {string} returns the hierarchicalFacet.rootPath or null as default
   */ _getHierarchicalRootPath: function(hierarchicalFacet) {
        return hierarchicalFacet.rootPath || null;
    },
    /**
   * Helper function to check if we show the parent level of the hierarchicalFacet
   * @private
   * @param  {object} hierarchicalFacet
   * @return {string} returns the hierarchicalFacet.showParentLevel or true as default
   */ _getHierarchicalShowParentLevel: function(hierarchicalFacet) {
        if (typeof hierarchicalFacet.showParentLevel === 'boolean') return hierarchicalFacet.showParentLevel;
        return true;
    },
    /**
   * Helper function to get the hierarchicalFacet by it's name
   * @param  {string} hierarchicalFacetName
   * @return {object} a hierarchicalFacet
   */ getHierarchicalFacetByName: function(hierarchicalFacetName) {
        return find(this.hierarchicalFacets, function(f) {
            return f.name === hierarchicalFacetName;
        });
    },
    /**
   * Get the current breadcrumb for a hierarchical facet, as an array
   * @param  {string} facetName Hierarchical facet name
   * @return {array.<string>} the path as an array of string
   */ getHierarchicalFacetBreadcrumb: function(facetName) {
        if (!this.isHierarchicalFacet(facetName)) return [];
        var refinement = this.getHierarchicalRefinement(facetName)[0];
        if (!refinement) return [];
        var separator = this._getHierarchicalFacetSeparator(this.getHierarchicalFacetByName(facetName));
        var path = refinement.split(separator);
        return path.map(function(part) {
            return part.trim();
        });
    },
    toString: function() {
        return JSON.stringify(this, null, 2);
    }
};
/**
 * Callback used for clearRefinement method
 * @callback SearchParameters.clearCallback
 * @param {OperatorList|FacetList} value the value of the filter
 * @param {string} key the current attribute name
 * @param {string} type `numeric`, `disjunctiveFacet`, `conjunctiveFacet`, `hierarchicalFacet` or `exclude`
 * depending on the type of facet
 * @return {boolean} `true` if the element should be removed. `false` otherwise.
 */ module.exports = SearchParameters;

},{"../functions/merge":"eGyc5","../functions/defaultsPure":"2BeUG","../functions/intersection":"iaaF4","../functions/find":"hBcv7","../functions/valToNumber":"jWUZB","../functions/omit":"l3IzD","../functions/objectHasKeys":"alqSr","../utils/isValidUserToken":"eZyse","./RefinementList":"5Zz04"}],"eGyc5":[function(require,module,exports) {
'use strict';
function clone(value) {
    if (typeof value === 'object' && value !== null) return _merge(Array.isArray(value) ? [] : {}, value);
    return value;
}
function isObjectOrArrayOrFunction(value) {
    return typeof value === 'function' || Array.isArray(value) || Object.prototype.toString.call(value) === '[object Object]';
}
function _merge(target, source) {
    if (target === source) return target;
    for(var key in source){
        if (!Object.prototype.hasOwnProperty.call(source, key) || key === '__proto__') continue;
        var sourceVal = source[key];
        var targetVal = target[key];
        if (typeof targetVal !== 'undefined' && typeof sourceVal === 'undefined') continue;
        if (isObjectOrArrayOrFunction(targetVal) && isObjectOrArrayOrFunction(sourceVal)) target[key] = _merge(targetVal, sourceVal);
        else target[key] = clone(sourceVal);
    }
    return target;
}
/**
 * This method is like Object.assign, but recursively merges own and inherited
 * enumerable keyed properties of source objects into the destination object.
 *
 * NOTE: this behaves like lodash/merge, but:
 * - does mutate functions if they are a source
 * - treats non-plain objects as plain
 * - does not work for circular objects
 * - treats sparse arrays as sparse
 * - does not convert Array-like objects (Arguments, NodeLists, etc.) to arrays
 *
 * @param {Object} object The destination object.
 * @param {...Object} [sources] The source objects.
 * @returns {Object} Returns `object`.
 */ function merge(target) {
    if (!isObjectOrArrayOrFunction(target)) target = {};
    for(var i = 1, l = arguments.length; i < l; i++){
        var source = arguments[i];
        if (isObjectOrArrayOrFunction(source)) _merge(target, source);
    }
    return target;
}
module.exports = merge;

},{}],"2BeUG":[function(require,module,exports) {
'use strict';
// NOTE: this behaves like lodash/defaults, but doesn't mutate the target
// it also preserve keys order
module.exports = function defaultsPure() {
    var sources = Array.prototype.slice.call(arguments);
    return sources.reduceRight(function(acc, source) {
        Object.keys(Object(source)).forEach(function(key) {
            if (source[key] === undefined) return;
            if (acc[key] !== undefined) // remove if already added, so that we can add it in correct order
            delete acc[key];
            acc[key] = source[key];
        });
        return acc;
    }, {});
};

},{}],"iaaF4":[function(require,module,exports) {
'use strict';
function intersection(arr1, arr2) {
    return arr1.filter(function(value, index) {
        return arr2.indexOf(value) > -1 && arr1.indexOf(value) === index /* skips duplicates */ ;
    });
}
module.exports = intersection;

},{}],"hBcv7":[function(require,module,exports) {
'use strict';
// @MAJOR can be replaced by native Array#find when we change support
module.exports = function find(array, comparator) {
    if (!Array.isArray(array)) return undefined;
    for(var i = 0; i < array.length; i++){
        if (comparator(array[i])) return array[i];
    }
};

},{}],"jWUZB":[function(require,module,exports) {
'use strict';
function valToNumber(v) {
    if (typeof v === 'number') return v;
    else if (typeof v === 'string') return parseFloat(v);
    else if (Array.isArray(v)) return v.map(valToNumber);
    throw new Error('The value should be a number, a parsable string or an array of those.');
}
module.exports = valToNumber;

},{}],"l3IzD":[function(require,module,exports) {
'use strict';
// https://github.com/babel/babel/blob/3aaafae053fa75febb3aa45d45b6f00646e30ba4/packages/babel-helpers/src/helpers.js#L604-L620
function _objectWithoutPropertiesLoose(source, excluded) {
    if (source === null) return {};
    var target = {};
    var sourceKeys = Object.keys(source);
    var key;
    var i;
    for(i = 0; i < sourceKeys.length; i++){
        key = sourceKeys[i];
        if (excluded.indexOf(key) >= 0) continue;
        target[key] = source[key];
    }
    return target;
}
module.exports = _objectWithoutPropertiesLoose;

},{}],"alqSr":[function(require,module,exports) {
'use strict';
function objectHasKeys(obj) {
    return obj && Object.keys(obj).length > 0;
}
module.exports = objectHasKeys;

},{}],"eZyse":[function(require,module,exports) {
'use strict';
module.exports = function isValidUserToken(userToken) {
    if (userToken === null) return false;
    return /^[a-zA-Z0-9_-]{1,64}$/.test(userToken);
};

},{}],"5Zz04":[function(require,module,exports) {
'use strict';
/**
 * Functions to manipulate refinement lists
 *
 * The RefinementList is not formally defined through a prototype but is based
 * on a specific structure.
 *
 * @module SearchParameters.refinementList
 *
 * @typedef {string[]} SearchParameters.refinementList.Refinements
 * @typedef {Object.<string, SearchParameters.refinementList.Refinements>} SearchParameters.refinementList.RefinementList
 */ var defaultsPure = require('../functions/defaultsPure');
var omit = require('../functions/omit');
var objectHasKeys = require('../functions/objectHasKeys');
var lib = {
    /**
   * Adds a refinement to a RefinementList
   * @param {RefinementList} refinementList the initial list
   * @param {string} attribute the attribute to refine
   * @param {string} value the value of the refinement, if the value is not a string it will be converted
   * @return {RefinementList} a new and updated refinement list
   */ addRefinement: function addRefinement(refinementList, attribute, value) {
        if (lib.isRefined(refinementList, attribute, value)) return refinementList;
        var valueAsString = '' + value;
        var facetRefinement = !refinementList[attribute] ? [
            valueAsString
        ] : refinementList[attribute].concat(valueAsString);
        var mod = {};
        mod[attribute] = facetRefinement;
        return defaultsPure({}, mod, refinementList);
    },
    /**
   * Removes refinement(s) for an attribute:
   *  - if the value is specified removes the refinement for the value on the attribute
   *  - if no value is specified removes all the refinements for this attribute
   * @param {RefinementList} refinementList the initial list
   * @param {string} attribute the attribute to refine
   * @param {string} [value] the value of the refinement
   * @return {RefinementList} a new and updated refinement lst
   */ removeRefinement: function removeRefinement(refinementList, attribute, value) {
        if (value === undefined) // we use the "filter" form of clearRefinement, since it leaves empty values as-is
        // the form with a string will remove the attribute completely
        return lib.clearRefinement(refinementList, function(v, f) {
            return attribute === f;
        });
        var valueAsString = '' + value;
        return lib.clearRefinement(refinementList, function(v, f) {
            return attribute === f && valueAsString === v;
        });
    },
    /**
   * Toggles the refinement value for an attribute.
   * @param {RefinementList} refinementList the initial list
   * @param {string} attribute the attribute to refine
   * @param {string} value the value of the refinement
   * @return {RefinementList} a new and updated list
   */ toggleRefinement: function toggleRefinement(refinementList, attribute, value) {
        if (value === undefined) throw new Error('toggleRefinement should be used with a value');
        if (lib.isRefined(refinementList, attribute, value)) return lib.removeRefinement(refinementList, attribute, value);
        return lib.addRefinement(refinementList, attribute, value);
    },
    /**
   * Clear all or parts of a RefinementList. Depending on the arguments, three
   * kinds of behavior can happen:
   *  - if no attribute is provided: clears the whole list
   *  - if an attribute is provided as a string: clears the list for the specific attribute
   *  - if an attribute is provided as a function: discards the elements for which the function returns true
   * @param {RefinementList} refinementList the initial list
   * @param {string} [attribute] the attribute or function to discard
   * @param {string} [refinementType] optional parameter to give more context to the attribute function
   * @return {RefinementList} a new and updated refinement list
   */ clearRefinement: function clearRefinement(refinementList, attribute, refinementType) {
        if (attribute === undefined) {
            if (!objectHasKeys(refinementList)) return refinementList;
            return {};
        } else if (typeof attribute === 'string') return omit(refinementList, [
            attribute
        ]);
        else if (typeof attribute === 'function') {
            var hasChanged = false;
            var newRefinementList = Object.keys(refinementList).reduce(function(memo, key) {
                var values = refinementList[key] || [];
                var facetList = values.filter(function(value) {
                    return !attribute(value, key, refinementType);
                });
                if (facetList.length !== values.length) hasChanged = true;
                memo[key] = facetList;
                return memo;
            }, {});
            if (hasChanged) return newRefinementList;
            return refinementList;
        }
    },
    /**
   * Test if the refinement value is used for the attribute. If no refinement value
   * is provided, test if the refinementList contains any refinement for the
   * given attribute.
   * @param {RefinementList} refinementList the list of refinement
   * @param {string} attribute name of the attribute
   * @param {string} [refinementValue] value of the filter/refinement
   * @return {boolean}
   */ isRefined: function isRefined(refinementList, attribute, refinementValue) {
        var containsRefinements = !!refinementList[attribute] && refinementList[attribute].length > 0;
        if (refinementValue === undefined || !containsRefinements) return containsRefinements;
        var refinementValueAsString = '' + refinementValue;
        return refinementList[attribute].indexOf(refinementValueAsString) !== -1;
    }
};
module.exports = lib;

},{"../functions/defaultsPure":"2BeUG","../functions/omit":"l3IzD","../functions/objectHasKeys":"alqSr"}],"lUGU6":[function(require,module,exports) {
'use strict';
var merge = require('../functions/merge');
var defaultsPure = require('../functions/defaultsPure');
var orderBy = require('../functions/orderBy');
var compact = require('../functions/compact');
var find = require('../functions/find');
var findIndex = require('../functions/findIndex');
var formatSort = require('../functions/formatSort');
var fv = require('../functions/escapeFacetValue');
var escapeFacetValue = fv.escapeFacetValue;
var unescapeFacetValue = fv.unescapeFacetValue;
var generateHierarchicalTree = require('./generate-hierarchical-tree');
/**
 * @typedef SearchResults.Facet
 * @type {object}
 * @property {string} name name of the attribute in the record
 * @property {object} data the faceting data: value, number of entries
 * @property {object} stats undefined unless facet_stats is retrieved from algolia
 */ /**
 * @typedef SearchResults.HierarchicalFacet
 * @type {object}
 * @property {string} name name of the current value given the hierarchical level, trimmed.
 * If root node, you get the facet name
 * @property {number} count number of objects matching this hierarchical value
 * @property {string} path the current hierarchical value full path
 * @property {boolean} isRefined `true` if the current value was refined, `false` otherwise
 * @property {HierarchicalFacet[]} data sub values for the current level
 */ /**
 * @typedef SearchResults.FacetValue
 * @type {object}
 * @property {string} name the facet value itself
 * @property {number} count times this facet appears in the results
 * @property {boolean} isRefined is the facet currently selected
 * @property {boolean} isExcluded is the facet currently excluded (only for conjunctive facets)
 */ /**
 * @typedef Refinement
 * @type {object}
 * @property {string} type the type of filter used:
 * `numeric`, `facet`, `exclude`, `disjunctive`, `hierarchical`
 * @property {string} attributeName name of the attribute used for filtering
 * @property {string} name the value of the filter
 * @property {number} numericValue the value as a number. Only for numeric filters.
 * @property {string} operator the operator used. Only for numeric filters.
 * @property {number} count the number of computed hits for this filter. Only on facets.
 * @property {boolean} exhaustive if the count is exhaustive
 */ /**
 * @param {string[]} attributes
 */ function getIndices(attributes) {
    var indices = {};
    attributes.forEach(function(val, idx) {
        indices[val] = idx;
    });
    return indices;
}
function assignFacetStats(dest, facetStats, key) {
    if (facetStats && facetStats[key]) dest.stats = facetStats[key];
}
/**
 * @typedef {Object} HierarchicalFacet
 * @property {string} name
 * @property {string[]} attributes
 */ /**
 * @param {HierarchicalFacet[]} hierarchicalFacets
 * @param {string} hierarchicalAttributeName
 */ function findMatchingHierarchicalFacetFromAttributeName(hierarchicalFacets, hierarchicalAttributeName) {
    return find(hierarchicalFacets, function facetKeyMatchesAttribute(hierarchicalFacet) {
        var facetNames = hierarchicalFacet.attributes || [];
        return facetNames.indexOf(hierarchicalAttributeName) > -1;
    });
}
/*eslint-disable */ /**
 * Constructor for SearchResults
 * @class
 * @classdesc SearchResults contains the results of a query to Algolia using the
 * {@link AlgoliaSearchHelper}.
 * @param {SearchParameters} state state that led to the response
 * @param {array.<object>} results the results from algolia client
 * @example <caption>SearchResults of the first query in
 * <a href="http://demos.algolia.com/instant-search-demo">the instant search demo</a></caption>
{
   "hitsPerPage": 10,
   "processingTimeMS": 2,
   "facets": [
      {
         "name": "type",
         "data": {
            "HardGood": 6627,
            "BlackTie": 550,
            "Music": 665,
            "Software": 131,
            "Game": 456,
            "Movie": 1571
         },
         "exhaustive": false
      },
      {
         "exhaustive": false,
         "data": {
            "Free shipping": 5507
         },
         "name": "shipping"
      }
  ],
   "hits": [
      {
         "thumbnailImage": "http://img.bbystatic.com/BestBuy_US/images/products/1688/1688832_54x108_s.gif",
         "_highlightResult": {
            "shortDescription": {
               "matchLevel": "none",
               "value": "Safeguard your PC, Mac, Android and iOS devices with comprehensive Internet protection",
               "matchedWords": []
            },
            "category": {
               "matchLevel": "none",
               "value": "Computer Security Software",
               "matchedWords": []
            },
            "manufacturer": {
               "matchedWords": [],
               "value": "Webroot",
               "matchLevel": "none"
            },
            "name": {
               "value": "Webroot SecureAnywhere Internet Security (3-Device) (1-Year Subscription) - Mac/Windows",
               "matchedWords": [],
               "matchLevel": "none"
            }
         },
         "image": "http://img.bbystatic.com/BestBuy_US/images/products/1688/1688832_105x210_sc.jpg",
         "shipping": "Free shipping",
         "bestSellingRank": 4,
         "shortDescription": "Safeguard your PC, Mac, Android and iOS devices with comprehensive Internet protection",
         "url": "http://www.bestbuy.com/site/webroot-secureanywhere-internet-security-3-devi…d=1219060687969&skuId=1688832&cmp=RMX&ky=2d3GfEmNIzjA0vkzveHdZEBgpPCyMnLTJ",
         "name": "Webroot SecureAnywhere Internet Security (3-Device) (1-Year Subscription) - Mac/Windows",
         "category": "Computer Security Software",
         "salePrice_range": "1 - 50",
         "objectID": "1688832",
         "type": "Software",
         "customerReviewCount": 5980,
         "salePrice": 49.99,
         "manufacturer": "Webroot"
      },
      ....
  ],
   "nbHits": 10000,
   "disjunctiveFacets": [
      {
         "exhaustive": false,
         "data": {
            "5": 183,
            "12": 112,
            "7": 149,
            ...
         },
         "name": "customerReviewCount",
         "stats": {
            "max": 7461,
            "avg": 157.939,
            "min": 1
         }
      },
      {
         "data": {
            "Printer Ink": 142,
            "Wireless Speakers": 60,
            "Point & Shoot Cameras": 48,
            ...
         },
         "name": "category",
         "exhaustive": false
      },
      {
         "exhaustive": false,
         "data": {
            "> 5000": 2,
            "1 - 50": 6524,
            "501 - 2000": 566,
            "201 - 500": 1501,
            "101 - 200": 1360,
            "2001 - 5000": 47
         },
         "name": "salePrice_range"
      },
      {
         "data": {
            "Dynex™": 202,
            "Insignia™": 230,
            "PNY": 72,
            ...
         },
         "name": "manufacturer",
         "exhaustive": false
      }
  ],
   "query": "",
   "nbPages": 100,
   "page": 0,
   "index": "bestbuy"
}
 **/ /*eslint-enable */ function SearchResults(state, results, options) {
    var mainSubResponse = results[0];
    this._rawResults = results;
    var self = this;
    // https://www.algolia.com/doc/api-reference/api-methods/search/#response
    Object.keys(mainSubResponse).forEach(function(key) {
        self[key] = mainSubResponse[key];
    });
    // Make every key of the result options reachable from the instance
    Object.keys(options || {}).forEach(function(key) {
        self[key] = options[key];
    });
    /**
   * query used to generate the results
   * @name query
   * @member {string}
   * @memberof SearchResults
   * @instance
   */ /**
   * The query as parsed by the engine given all the rules.
   * @name parsedQuery
   * @member {string}
   * @memberof SearchResults
   * @instance
   */ /**
   * all the records that match the search parameters. Each record is
   * augmented with a new attribute `_highlightResult`
   * which is an object keyed by attribute and with the following properties:
   *  - `value` : the value of the facet highlighted (html)
   *  - `matchLevel`: full, partial or none depending on how the query terms match
   * @name hits
   * @member {object[]}
   * @memberof SearchResults
   * @instance
   */ /**
   * index where the results come from
   * @name index
   * @member {string}
   * @memberof SearchResults
   * @instance
   */ /**
   * number of hits per page requested
   * @name hitsPerPage
   * @member {number}
   * @memberof SearchResults
   * @instance
   */ /**
   * total number of hits of this query on the index
   * @name nbHits
   * @member {number}
   * @memberof SearchResults
   * @instance
   */ /**
   * total number of pages with respect to the number of hits per page and the total number of hits
   * @name nbPages
   * @member {number}
   * @memberof SearchResults
   * @instance
   */ /**
   * current page
   * @name page
   * @member {number}
   * @memberof SearchResults
   * @instance
   */ /**
   * The position if the position was guessed by IP.
   * @name aroundLatLng
   * @member {string}
   * @memberof SearchResults
   * @instance
   * @example "48.8637,2.3615",
   */ /**
   * The radius computed by Algolia.
   * @name automaticRadius
   * @member {string}
   * @memberof SearchResults
   * @instance
   * @example "126792922",
   */ /**
   * String identifying the server used to serve this request.
   *
   * getRankingInfo needs to be set to `true` for this to be returned
   *
   * @name serverUsed
   * @member {string}
   * @memberof SearchResults
   * @instance
   * @example "c7-use-2.algolia.net",
   */ /**
   * Boolean that indicates if the computation of the counts did time out.
   * @deprecated
   * @name timeoutCounts
   * @member {boolean}
   * @memberof SearchResults
   * @instance
   */ /**
   * Boolean that indicates if the computation of the hits did time out.
   * @deprecated
   * @name timeoutHits
   * @member {boolean}
   * @memberof SearchResults
   * @instance
   */ /**
   * True if the counts of the facets is exhaustive
   * @name exhaustiveFacetsCount
   * @member {boolean}
   * @memberof SearchResults
   * @instance
   */ /**
   * True if the number of hits is exhaustive
   * @name exhaustiveNbHits
   * @member {boolean}
   * @memberof SearchResults
   * @instance
   */ /**
   * Contains the userData if they are set by a [query rule](https://www.algolia.com/doc/guides/query-rules/query-rules-overview/).
   * @name userData
   * @member {object[]}
   * @memberof SearchResults
   * @instance
   */ /**
   * queryID is the unique identifier of the query used to generate the current search results.
   * This value is only available if the `clickAnalytics` search parameter is set to `true`.
   * @name queryID
   * @member {string}
   * @memberof SearchResults
   * @instance
   */ /**
   * sum of the processing time of all the queries
   * @member {number}
   */ this.processingTimeMS = results.reduce(function(sum, result) {
        return result.processingTimeMS === undefined ? sum : sum + result.processingTimeMS;
    }, 0);
    /**
   * disjunctive facets results
   * @member {SearchResults.Facet[]}
   */ this.disjunctiveFacets = [];
    /**
   * disjunctive facets results
   * @member {SearchResults.HierarchicalFacet[]}
   */ this.hierarchicalFacets = state.hierarchicalFacets.map(function initFutureTree() {
        return [];
    });
    /**
   * other facets results
   * @member {SearchResults.Facet[]}
   */ this.facets = [];
    var disjunctiveFacets = state.getRefinedDisjunctiveFacets();
    var facetsIndices = getIndices(state.facets);
    var disjunctiveFacetsIndices = getIndices(state.disjunctiveFacets);
    var nextDisjunctiveResult = 1;
    // Since we send request only for disjunctive facets that have been refined,
    // we get the facets information from the first, general, response.
    var mainFacets = mainSubResponse.facets || {};
    Object.keys(mainFacets).forEach(function(facetKey) {
        var facetValueObject = mainFacets[facetKey];
        var hierarchicalFacet = findMatchingHierarchicalFacetFromAttributeName(state.hierarchicalFacets, facetKey);
        if (hierarchicalFacet) {
            // Place the hierarchicalFacet data at the correct index depending on
            // the attributes order that was defined at the helper initialization
            var facetIndex = hierarchicalFacet.attributes.indexOf(facetKey);
            var idxAttributeName = findIndex(state.hierarchicalFacets, function(f) {
                return f.name === hierarchicalFacet.name;
            });
            self.hierarchicalFacets[idxAttributeName][facetIndex] = {
                attribute: facetKey,
                data: facetValueObject,
                exhaustive: mainSubResponse.exhaustiveFacetsCount
            };
        } else {
            var isFacetDisjunctive = state.disjunctiveFacets.indexOf(facetKey) !== -1;
            var isFacetConjunctive = state.facets.indexOf(facetKey) !== -1;
            var position;
            if (isFacetDisjunctive) {
                position = disjunctiveFacetsIndices[facetKey];
                self.disjunctiveFacets[position] = {
                    name: facetKey,
                    data: facetValueObject,
                    exhaustive: mainSubResponse.exhaustiveFacetsCount
                };
                assignFacetStats(self.disjunctiveFacets[position], mainSubResponse.facets_stats, facetKey);
            }
            if (isFacetConjunctive) {
                position = facetsIndices[facetKey];
                self.facets[position] = {
                    name: facetKey,
                    data: facetValueObject,
                    exhaustive: mainSubResponse.exhaustiveFacetsCount
                };
                assignFacetStats(self.facets[position], mainSubResponse.facets_stats, facetKey);
            }
        }
    });
    // Make sure we do not keep holes within the hierarchical facets
    this.hierarchicalFacets = compact(this.hierarchicalFacets);
    // aggregate the refined disjunctive facets
    disjunctiveFacets.forEach(function(disjunctiveFacet) {
        var result = results[nextDisjunctiveResult];
        var facets = result && result.facets ? result.facets : {};
        var hierarchicalFacet = state.getHierarchicalFacetByName(disjunctiveFacet);
        // There should be only item in facets.
        Object.keys(facets).forEach(function(dfacet) {
            var facetResults = facets[dfacet];
            var position;
            if (hierarchicalFacet) {
                position = findIndex(state.hierarchicalFacets, function(f) {
                    return f.name === hierarchicalFacet.name;
                });
                var attributeIndex = findIndex(self.hierarchicalFacets[position], function(f) {
                    return f.attribute === dfacet;
                });
                // previous refinements and no results so not able to find it
                if (attributeIndex === -1) return;
                self.hierarchicalFacets[position][attributeIndex].data = merge({}, self.hierarchicalFacets[position][attributeIndex].data, facetResults);
            } else {
                position = disjunctiveFacetsIndices[dfacet];
                var dataFromMainRequest = mainSubResponse.facets && mainSubResponse.facets[dfacet] || {};
                self.disjunctiveFacets[position] = {
                    name: dfacet,
                    data: defaultsPure({}, facetResults, dataFromMainRequest),
                    exhaustive: result.exhaustiveFacetsCount
                };
                assignFacetStats(self.disjunctiveFacets[position], result.facets_stats, dfacet);
                if (state.disjunctiveFacetsRefinements[dfacet]) state.disjunctiveFacetsRefinements[dfacet].forEach(function(refinementValue) {
                    // add the disjunctive refinements if it is no more retrieved
                    if (!self.disjunctiveFacets[position].data[refinementValue] && state.disjunctiveFacetsRefinements[dfacet].indexOf(unescapeFacetValue(refinementValue)) > -1) self.disjunctiveFacets[position].data[refinementValue] = 0;
                });
            }
        });
        nextDisjunctiveResult++;
    });
    // if we have some root level values for hierarchical facets, merge them
    state.getRefinedHierarchicalFacets().forEach(function(refinedFacet) {
        var hierarchicalFacet = state.getHierarchicalFacetByName(refinedFacet);
        var separator = state._getHierarchicalFacetSeparator(hierarchicalFacet);
        var currentRefinement = state.getHierarchicalRefinement(refinedFacet);
        // if we are already at a root refinement (or no refinement at all), there is no
        // root level values request
        if (currentRefinement.length === 0 || currentRefinement[0].split(separator).length < 2) return;
        var result = results[nextDisjunctiveResult];
        var facets = result && result.facets ? result.facets : {};
        Object.keys(facets).forEach(function(dfacet) {
            var facetResults = facets[dfacet];
            var position = findIndex(state.hierarchicalFacets, function(f) {
                return f.name === hierarchicalFacet.name;
            });
            var attributeIndex = findIndex(self.hierarchicalFacets[position], function(f) {
                return f.attribute === dfacet;
            });
            // previous refinements and no results so not able to find it
            if (attributeIndex === -1) return;
            // when we always get root levels, if the hits refinement is `beers > IPA` (count: 5),
            // then the disjunctive values will be `beers` (count: 100),
            // but we do not want to display
            //   | beers (100)
            //     > IPA (5)
            // We want
            //   | beers (5)
            //     > IPA (5)
            var defaultData = {};
            if (currentRefinement.length > 0) {
                var root = currentRefinement[0].split(separator)[0];
                defaultData[root] = self.hierarchicalFacets[position][attributeIndex].data[root];
            }
            self.hierarchicalFacets[position][attributeIndex].data = defaultsPure(defaultData, facetResults, self.hierarchicalFacets[position][attributeIndex].data);
        });
        nextDisjunctiveResult++;
    });
    // add the excludes
    Object.keys(state.facetsExcludes).forEach(function(facetName) {
        var excludes = state.facetsExcludes[facetName];
        var position = facetsIndices[facetName];
        self.facets[position] = {
            name: facetName,
            data: mainSubResponse.facets[facetName],
            exhaustive: mainSubResponse.exhaustiveFacetsCount
        };
        excludes.forEach(function(facetValue) {
            self.facets[position] = self.facets[position] || {
                name: facetName
            };
            self.facets[position].data = self.facets[position].data || {};
            self.facets[position].data[facetValue] = 0;
        });
    });
    /**
   * @type {Array}
   */ this.hierarchicalFacets = this.hierarchicalFacets.map(generateHierarchicalTree(state));
    /**
   * @type {Array}
   */ this.facets = compact(this.facets);
    /**
   * @type {Array}
   */ this.disjunctiveFacets = compact(this.disjunctiveFacets);
    this._state = state;
}
/**
 * Get a facet object with its name
 * @deprecated
 * @param {string} name name of the faceted attribute
 * @return {SearchResults.Facet} the facet object
 */ SearchResults.prototype.getFacetByName = function(name) {
    function predicate(facet) {
        return facet.name === name;
    }
    return find(this.facets, predicate) || find(this.disjunctiveFacets, predicate) || find(this.hierarchicalFacets, predicate);
};
/**
 * Get the facet values of a specified attribute from a SearchResults object.
 * @private
 * @param {SearchResults} results the search results to search in
 * @param {string} attribute name of the faceted attribute to search for
 * @return {array|object} facet values. For the hierarchical facets it is an object.
 */ function extractNormalizedFacetValues(results, attribute) {
    function predicate(facet) {
        return facet.name === attribute;
    }
    if (results._state.isConjunctiveFacet(attribute)) {
        var facet1 = find(results.facets, predicate);
        if (!facet1) return [];
        return Object.keys(facet1.data).map(function(name) {
            var value = escapeFacetValue(name);
            return {
                name: name,
                escapedValue: value,
                count: facet1.data[name],
                isRefined: results._state.isFacetRefined(attribute, value),
                isExcluded: results._state.isExcludeRefined(attribute, name)
            };
        });
    } else if (results._state.isDisjunctiveFacet(attribute)) {
        var disjunctiveFacet = find(results.disjunctiveFacets, predicate);
        if (!disjunctiveFacet) return [];
        return Object.keys(disjunctiveFacet.data).map(function(name) {
            var value = escapeFacetValue(name);
            return {
                name: name,
                escapedValue: value,
                count: disjunctiveFacet.data[name],
                isRefined: results._state.isDisjunctiveFacetRefined(attribute, value)
            };
        });
    } else if (results._state.isHierarchicalFacet(attribute)) return find(results.hierarchicalFacets, predicate);
}
/**
 * Sort nodes of a hierarchical or disjunctive facet results
 * @private
 * @param {function} sortFn
 * @param {HierarchicalFacet|Array} node node upon which we want to apply the sort
 * @param {string[]} names attribute names
 * @param {number} [level=0] current index in the names array
 */ function recSort(sortFn, node, names, level) {
    level = level || 0;
    if (Array.isArray(node)) return sortFn(node, names[level]);
    if (!node.data || node.data.length === 0) return node;
    var children = node.data.map(function(childNode) {
        return recSort(sortFn, childNode, names, level + 1);
    });
    var sortedChildren = sortFn(children, names[level]);
    var newNode = defaultsPure({
        data: sortedChildren
    }, node);
    return newNode;
}
SearchResults.DEFAULT_SORT = [
    'isRefined:desc',
    'count:desc',
    'name:asc'
];
function vanillaSortFn(order, data) {
    return data.sort(order);
}
/**
 * @typedef FacetOrdering
 * @type {Object}
 * @property {string[]} [order]
 * @property {'count' | 'alpha' | 'hidden'} [sortRemainingBy]
 */ /**
 * Sorts facet arrays via their facet ordering
 * @param {Array} facetValues the values
 * @param {FacetOrdering} facetOrdering the ordering
 * @returns {Array}
 */ function sortViaFacetOrdering(facetValues, facetOrdering) {
    var orderedFacets = [];
    var remainingFacets = [];
    var order = facetOrdering.order || [];
    /**
   * an object with the keys being the values in order, the values their index:
   * ['one', 'two'] -> { one: 0, two: 1 }
   */ var reverseOrder = order.reduce(function(acc, name, i) {
        acc[name] = i;
        return acc;
    }, {});
    facetValues.forEach(function(item) {
        // hierarchical facets get sorted using their raw name
        var name = item.path || item.name;
        if (reverseOrder[name] !== undefined) orderedFacets[reverseOrder[name]] = item;
        else remainingFacets.push(item);
    });
    orderedFacets = orderedFacets.filter(function(facet) {
        return facet;
    });
    var sortRemainingBy = facetOrdering.sortRemainingBy;
    var ordering;
    if (sortRemainingBy === 'hidden') return orderedFacets;
    else if (sortRemainingBy === 'alpha') ordering = [
        [
            'path',
            'name'
        ],
        [
            'asc',
            'asc'
        ]
    ];
    else ordering = [
        [
            'count'
        ],
        [
            'desc'
        ]
    ];
    return orderedFacets.concat(orderBy(remainingFacets, ordering[0], ordering[1]));
}
/**
 * @param {SearchResults} results the search results class
 * @param {string} attribute the attribute to retrieve ordering of
 * @returns {FacetOrdering=}
 */ function getFacetOrdering(results, attribute) {
    return results.renderingContent && results.renderingContent.facetOrdering && results.renderingContent.facetOrdering.values && results.renderingContent.facetOrdering.values[attribute];
}
/**
 * Get a the list of values for a given facet attribute. Those values are sorted
 * refinement first, descending count (bigger value on top), and name ascending
 * (alphabetical order). The sort formula can overridden using either string based
 * predicates or a function.
 *
 * This method will return all the values returned by the Algolia engine plus all
 * the values already refined. This means that it can happen that the
 * `maxValuesPerFacet` [configuration](https://www.algolia.com/doc/rest-api/search#param-maxValuesPerFacet)
 * might not be respected if you have facet values that are already refined.
 * @param {string} attribute attribute name
 * @param {object} opts configuration options.
 * @param {boolean} [opts.facetOrdering]
 * Force the use of facetOrdering from the result if a sortBy is present. If
 * sortBy isn't present, facetOrdering will be used automatically.
 * @param {Array.<string> | function} opts.sortBy
 * When using strings, it consists of
 * the name of the [FacetValue](#SearchResults.FacetValue) or the
 * [HierarchicalFacet](#SearchResults.HierarchicalFacet) attributes with the
 * order (`asc` or `desc`). For example to order the value by count, the
 * argument would be `['count:asc']`.
 *
 * If only the attribute name is specified, the ordering defaults to the one
 * specified in the default value for this attribute.
 *
 * When not specified, the order is
 * ascending.  This parameter can also be a function which takes two facet
 * values and should return a number, 0 if equal, 1 if the first argument is
 * bigger or -1 otherwise.
 *
 * The default value for this attribute `['isRefined:desc', 'count:desc', 'name:asc']`
 * @return {FacetValue[]|HierarchicalFacet|undefined} depending on the type of facet of
 * the attribute requested (hierarchical, disjunctive or conjunctive)
 * @example
 * helper.on('result', function(event){
 *   //get values ordered only by name ascending using the string predicate
 *   event.results.getFacetValues('city', {sortBy: ['name:asc']});
 *   //get values  ordered only by count ascending using a function
 *   event.results.getFacetValues('city', {
 *     // this is equivalent to ['count:asc']
 *     sortBy: function(a, b) {
 *       if (a.count === b.count) return 0;
 *       if (a.count > b.count)   return 1;
 *       if (b.count > a.count)   return -1;
 *     }
 *   });
 * });
 */ SearchResults.prototype.getFacetValues = function(attribute, opts) {
    var facetValues = extractNormalizedFacetValues(this, attribute);
    if (!facetValues) return undefined;
    var options = defaultsPure({}, opts, {
        sortBy: SearchResults.DEFAULT_SORT,
        // if no sortBy is given, attempt to sort based on facetOrdering
        // if it is given, we still allow to sort via facet ordering first
        facetOrdering: !(opts && opts.sortBy)
    });
    var results = this;
    var attributes;
    if (Array.isArray(facetValues)) attributes = [
        attribute
    ];
    else {
        var config = results._state.getHierarchicalFacetByName(facetValues.name);
        attributes = config.attributes;
    }
    return recSort(function(data, facetName) {
        if (options.facetOrdering) {
            var facetOrdering = getFacetOrdering(results, facetName);
            if (Boolean(facetOrdering)) return sortViaFacetOrdering(data, facetOrdering);
        }
        if (Array.isArray(options.sortBy)) {
            var order = formatSort(options.sortBy, SearchResults.DEFAULT_SORT);
            return orderBy(data, order[0], order[1]);
        } else if (typeof options.sortBy === 'function') return vanillaSortFn(options.sortBy, data);
        throw new Error("options.sortBy is optional but if defined it must be either an array of string (predicates) or a sorting function");
    }, facetValues, attributes);
};
/**
 * Returns the facet stats if attribute is defined and the facet contains some.
 * Otherwise returns undefined.
 * @param {string} attribute name of the faceted attribute
 * @return {object} The stats of the facet
 */ SearchResults.prototype.getFacetStats = function(attribute) {
    if (this._state.isConjunctiveFacet(attribute)) return getFacetStatsIfAvailable(this.facets, attribute);
    else if (this._state.isDisjunctiveFacet(attribute)) return getFacetStatsIfAvailable(this.disjunctiveFacets, attribute);
    return undefined;
};
/**
 * @typedef {Object} FacetListItem
 * @property {string} name
 */ /**
 * @param {FacetListItem[]} facetList (has more items, but enough for here)
 * @param {string} facetName
 */ function getFacetStatsIfAvailable(facetList, facetName) {
    var data = find(facetList, function(facet) {
        return facet.name === facetName;
    });
    return data && data.stats;
}
/**
 * Returns all refinements for all filters + tags. It also provides
 * additional information: count and exhaustiveness for each filter.
 *
 * See the [refinement type](#Refinement) for an exhaustive view of the available
 * data.
 *
 * Note that for a numeric refinement, results are grouped per operator, this
 * means that it will return responses for operators which are empty.
 *
 * @return {Array.<Refinement>} all the refinements
 */ SearchResults.prototype.getRefinements = function() {
    var state = this._state;
    var results = this;
    var res = [];
    Object.keys(state.facetsRefinements).forEach(function(attributeName) {
        state.facetsRefinements[attributeName].forEach(function(name) {
            res.push(getRefinement(state, 'facet', attributeName, name, results.facets));
        });
    });
    Object.keys(state.facetsExcludes).forEach(function(attributeName) {
        state.facetsExcludes[attributeName].forEach(function(name) {
            res.push(getRefinement(state, 'exclude', attributeName, name, results.facets));
        });
    });
    Object.keys(state.disjunctiveFacetsRefinements).forEach(function(attributeName) {
        state.disjunctiveFacetsRefinements[attributeName].forEach(function(name) {
            res.push(getRefinement(state, 'disjunctive', attributeName, name, results.disjunctiveFacets));
        });
    });
    Object.keys(state.hierarchicalFacetsRefinements).forEach(function(attributeName) {
        state.hierarchicalFacetsRefinements[attributeName].forEach(function(name) {
            res.push(getHierarchicalRefinement(state, attributeName, name, results.hierarchicalFacets));
        });
    });
    Object.keys(state.numericRefinements).forEach(function(attributeName) {
        var operators = state.numericRefinements[attributeName];
        Object.keys(operators).forEach(function(operator) {
            operators[operator].forEach(function(value) {
                res.push({
                    type: 'numeric',
                    attributeName: attributeName,
                    name: value,
                    numericValue: value,
                    operator: operator
                });
            });
        });
    });
    state.tagRefinements.forEach(function(name) {
        res.push({
            type: 'tag',
            attributeName: '_tags',
            name: name
        });
    });
    return res;
};
/**
 * @typedef {Object} Facet
 * @property {string} name
 * @property {Object} data
 * @property {boolean} exhaustive
 */ /**
 * @param {*} state
 * @param {*} type
 * @param {string} attributeName
 * @param {*} name
 * @param {Facet[]} resultsFacets
 */ function getRefinement(state, type, attributeName, name, resultsFacets) {
    var facet = find(resultsFacets, function(f) {
        return f.name === attributeName;
    });
    var count = facet && facet.data && facet.data[name] ? facet.data[name] : 0;
    var exhaustive = facet && facet.exhaustive || false;
    return {
        type: type,
        attributeName: attributeName,
        name: name,
        count: count,
        exhaustive: exhaustive
    };
}
/**
 * @param {*} state
 * @param {string} attributeName
 * @param {*} name
 * @param {Facet[]} resultsFacets
 */ function getHierarchicalRefinement(state, attributeName, name, resultsFacets) {
    var facetDeclaration = state.getHierarchicalFacetByName(attributeName);
    var separator = state._getHierarchicalFacetSeparator(facetDeclaration);
    var split = name.split(separator);
    var rootFacet = find(resultsFacets, function(facet) {
        return facet.name === attributeName;
    });
    var facet2 = split.reduce(function(intermediateFacet, part) {
        var newFacet = intermediateFacet && find(intermediateFacet.data, function(f) {
            return f.name === part;
        });
        return newFacet !== undefined ? newFacet : intermediateFacet;
    }, rootFacet);
    var count = facet2 && facet2.count || 0;
    var exhaustive = facet2 && facet2.exhaustive || false;
    var path = facet2 && facet2.path || '';
    return {
        type: 'hierarchical',
        attributeName: attributeName,
        name: path,
        count: count,
        exhaustive: exhaustive
    };
}
module.exports = SearchResults;

},{"../functions/merge":"eGyc5","../functions/defaultsPure":"2BeUG","../functions/orderBy":"kd35s","../functions/compact":"dFh0T","../functions/find":"hBcv7","../functions/findIndex":"fzLII","../functions/formatSort":"g3eEb","../functions/escapeFacetValue":"3r1Qc","./generate-hierarchical-tree":"9tLzD"}],"kd35s":[function(require,module,exports) {
'use strict';
function compareAscending(value, other) {
    if (value !== other) {
        var valIsDefined = value !== undefined;
        var valIsNull = value === null;
        var othIsDefined = other !== undefined;
        var othIsNull = other === null;
        if (!othIsNull && value > other || valIsNull && othIsDefined || !valIsDefined) return 1;
        if (!valIsNull && value < other || othIsNull && valIsDefined || !othIsDefined) return -1;
    }
    return 0;
}
/**
 * @param {Array<object>} collection object with keys in attributes
 * @param {Array<string>} iteratees attributes
 * @param {Array<string>} orders asc | desc
 */ function orderBy(collection, iteratees, orders) {
    if (!Array.isArray(collection)) return [];
    if (!Array.isArray(orders)) orders = [];
    var result = collection.map(function(value, index) {
        return {
            criteria: iteratees.map(function(iteratee) {
                return value[iteratee];
            }),
            index: index,
            value: value
        };
    });
    result.sort(function comparer(object, other) {
        var index = -1;
        while(++index < object.criteria.length){
            var res = compareAscending(object.criteria[index], other.criteria[index]);
            if (res) {
                if (index >= orders.length) return res;
                if (orders[index] === 'desc') return -res;
                return res;
            }
        }
        // This ensures a stable sort in V8 and other engines.
        // See https://bugs.chromium.org/p/v8/issues/detail?id=90 for more details.
        return object.index - other.index;
    });
    return result.map(function(res) {
        return res.value;
    });
}
module.exports = orderBy;

},{}],"dFh0T":[function(require,module,exports) {
'use strict';
module.exports = function compact(array) {
    if (!Array.isArray(array)) return [];
    return array.filter(Boolean);
};

},{}],"fzLII":[function(require,module,exports) {
'use strict';
// @MAJOR can be replaced by native Array#findIndex when we change support
module.exports = function find(array, comparator) {
    if (!Array.isArray(array)) return -1;
    for(var i = 0; i < array.length; i++){
        if (comparator(array[i])) return i;
    }
    return -1;
};

},{}],"g3eEb":[function(require,module,exports) {
'use strict';
var find = require('./find');
/**
 * Transform sort format from user friendly notation to lodash format
 * @param {string[]} sortBy array of predicate of the form "attribute:order"
 * @param {string[]} [defaults] array of predicate of the form "attribute:order"
 * @return {array.<string[]>} array containing 2 elements : attributes, orders
 */ module.exports = function formatSort(sortBy, defaults) {
    var defaultInstructions = (defaults || []).map(function(sort) {
        return sort.split(':');
    });
    return sortBy.reduce(function preparePredicate(out, sort) {
        var sortInstruction = sort.split(':');
        var matchingDefault = find(defaultInstructions, function(defaultInstruction) {
            return defaultInstruction[0] === sortInstruction[0];
        });
        if (sortInstruction.length > 1 || !matchingDefault) {
            out[0].push(sortInstruction[0]);
            out[1].push(sortInstruction[1]);
            return out;
        }
        out[0].push(matchingDefault[0]);
        out[1].push(matchingDefault[1]);
        return out;
    }, [
        [],
        []
    ]);
};

},{"./find":"hBcv7"}],"3r1Qc":[function(require,module,exports) {
'use strict';
/**
 * Replaces a leading - with \-
 * @private
 * @param {any} value the facet value to replace
 * @returns any
 */ function escapeFacetValue(value) {
    if (typeof value !== 'string') return value;
    return String(value).replace(/^-/, '\\-');
}
/**
 * Replaces a leading \- with -
 * @private
 * @param {any} value the escaped facet value
 * @returns any
 */ function unescapeFacetValue(value) {
    if (typeof value !== 'string') return value;
    return value.replace(/^\\-/, '-');
}
module.exports = {
    escapeFacetValue: escapeFacetValue,
    unescapeFacetValue: unescapeFacetValue
};

},{}],"9tLzD":[function(require,module,exports) {
'use strict';
module.exports = generateTrees;
var orderBy = require('../functions/orderBy');
var find = require('../functions/find');
var prepareHierarchicalFacetSortBy = require('../functions/formatSort');
var fv = require('../functions/escapeFacetValue');
var escapeFacetValue = fv.escapeFacetValue;
var unescapeFacetValue = fv.unescapeFacetValue;
function generateTrees(state) {
    return function generate(hierarchicalFacetResult, hierarchicalFacetIndex) {
        var hierarchicalFacet = state.hierarchicalFacets[hierarchicalFacetIndex];
        var hierarchicalFacetRefinement = state.hierarchicalFacetsRefinements[hierarchicalFacet.name] && state.hierarchicalFacetsRefinements[hierarchicalFacet.name][0] || '';
        var hierarchicalSeparator = state._getHierarchicalFacetSeparator(hierarchicalFacet);
        var hierarchicalRootPath = state._getHierarchicalRootPath(hierarchicalFacet);
        var hierarchicalShowParentLevel = state._getHierarchicalShowParentLevel(hierarchicalFacet);
        var sortBy = prepareHierarchicalFacetSortBy(state._getHierarchicalFacetSortBy(hierarchicalFacet));
        var rootExhaustive = hierarchicalFacetResult.every(function(facetResult) {
            return facetResult.exhaustive;
        });
        var generateTreeFn = generateHierarchicalTree(sortBy, hierarchicalSeparator, hierarchicalRootPath, hierarchicalShowParentLevel, hierarchicalFacetRefinement);
        var results = hierarchicalFacetResult;
        if (hierarchicalRootPath) results = hierarchicalFacetResult.slice(hierarchicalRootPath.split(hierarchicalSeparator).length);
        return results.reduce(generateTreeFn, {
            name: state.hierarchicalFacets[hierarchicalFacetIndex].name,
            count: null,
            isRefined: true,
            path: null,
            escapedValue: null,
            exhaustive: rootExhaustive,
            data: null
        });
    };
}
function generateHierarchicalTree(sortBy, hierarchicalSeparator, hierarchicalRootPath, hierarchicalShowParentLevel, currentRefinement) {
    return function generateTree(hierarchicalTree, hierarchicalFacetResult, currentHierarchicalLevel) {
        var parent = hierarchicalTree;
        if (currentHierarchicalLevel > 0) {
            var level = 0;
            parent = hierarchicalTree;
            while(level < currentHierarchicalLevel){
                /**
         * @type {object[]]} hierarchical data
         */ var data = parent && Array.isArray(parent.data) ? parent.data : [];
                parent = find(data, function(subtree) {
                    return subtree.isRefined;
                });
                level++;
            }
        }
        // we found a refined parent, let's add current level data under it
        if (parent) {
            // filter values in case an object has multiple categories:
            //   {
            //     categories: {
            //       level0: ['beers', 'bières'],
            //       level1: ['beers > IPA', 'bières > Belges']
            //     }
            //   }
            //
            // If parent refinement is `beers`, then we do not want to have `bières > Belges`
            // showing up
            var picked = Object.keys(hierarchicalFacetResult.data).map(function(facetValue) {
                return [
                    facetValue,
                    hierarchicalFacetResult.data[facetValue]
                ];
            }).filter(function(tuple) {
                var facetValue = tuple[0];
                return onlyMatchingTree(facetValue, parent.path || hierarchicalRootPath, currentRefinement, hierarchicalSeparator, hierarchicalRootPath, hierarchicalShowParentLevel);
            });
            parent.data = orderBy(picked.map(function(tuple) {
                var facetValue = tuple[0];
                var facetCount = tuple[1];
                return format(facetCount, facetValue, hierarchicalSeparator, unescapeFacetValue(currentRefinement), hierarchicalFacetResult.exhaustive);
            }), sortBy[0], sortBy[1]);
        }
        return hierarchicalTree;
    };
}
function onlyMatchingTree(facetValue, parentPath, currentRefinement, hierarchicalSeparator, hierarchicalRootPath, hierarchicalShowParentLevel) {
    // we want the facetValue is a child of hierarchicalRootPath
    if (hierarchicalRootPath && (facetValue.indexOf(hierarchicalRootPath) !== 0 || hierarchicalRootPath === facetValue)) return false;
    // we always want root levels (only when there is no prefix path)
    return !hierarchicalRootPath && facetValue.indexOf(hierarchicalSeparator) === -1 || hierarchicalRootPath && facetValue.split(hierarchicalSeparator).length - hierarchicalRootPath.split(hierarchicalSeparator).length === 1 || facetValue.indexOf(hierarchicalSeparator) === -1 && currentRefinement.indexOf(hierarchicalSeparator) === -1 || // currentRefinement is a child of the facet value
    currentRefinement.indexOf(facetValue) === 0 || facetValue.indexOf(parentPath + hierarchicalSeparator) === 0 && (hierarchicalShowParentLevel || facetValue.indexOf(currentRefinement) === 0);
}
function format(facetCount, facetValue, hierarchicalSeparator, currentRefinement, exhaustive) {
    var parts = facetValue.split(hierarchicalSeparator);
    return {
        name: parts[parts.length - 1].trim(),
        path: facetValue,
        escapedValue: escapeFacetValue(facetValue),
        count: facetCount,
        isRefined: currentRefinement === facetValue || currentRefinement.indexOf(facetValue + hierarchicalSeparator) === 0,
        exhaustive: exhaustive,
        data: null
    };
}

},{"../functions/orderBy":"kd35s","../functions/find":"hBcv7","../functions/formatSort":"g3eEb","../functions/escapeFacetValue":"3r1Qc"}],"6UDS7":[function(require,module,exports) {
'use strict';
var EventEmitter = require('@algolia/events');
var inherits = require('../functions/inherits');
/**
 * A DerivedHelper is a way to create sub requests to
 * Algolia from a main helper.
 * @class
 * @classdesc The DerivedHelper provides an event based interface for search callbacks:
 *  - search: when a search is triggered using the `search()` method.
 *  - result: when the response is retrieved from Algolia and is processed.
 *    This event contains a {@link SearchResults} object and the
 *    {@link SearchParameters} corresponding to this answer.
 */ function DerivedHelper(mainHelper, fn) {
    this.main = mainHelper;
    this.fn = fn;
    this.lastResults = null;
}
inherits(DerivedHelper, EventEmitter);
/**
 * Detach this helper from the main helper
 * @return {undefined}
 * @throws Error if the derived helper is already detached
 */ DerivedHelper.prototype.detach = function() {
    this.removeAllListeners();
    this.main.detachDerivedHelper(this);
};
DerivedHelper.prototype.getModifiedState = function(parameters) {
    return this.fn(parameters);
};
module.exports = DerivedHelper;

},{"@algolia/events":"euNDO","../functions/inherits":"a0E30"}],"euNDO":[function(require,module,exports) {
// Copyright Joyent, Inc. and other Node contributors.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit
// persons to whom the Software is furnished to do so, subject to the
// following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
// NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
// USE OR OTHER DEALINGS IN THE SOFTWARE.
function EventEmitter() {
    this._events = this._events || {};
    this._maxListeners = this._maxListeners || undefined;
}
module.exports = EventEmitter;
// Backwards-compat with node 0.10.x
// EventEmitter.EventEmitter = EventEmitter;
EventEmitter.prototype._events = undefined;
EventEmitter.prototype._maxListeners = undefined;
// By default EventEmitters will print a warning if more than 10 listeners are
// added to it. This is a useful default which helps finding memory leaks.
EventEmitter.defaultMaxListeners = 10;
// Obviously not all Emitters should be limited to 10. This function allows
// that to be increased. Set to zero for unlimited.
EventEmitter.prototype.setMaxListeners = function(n) {
    if (!isNumber(n) || n < 0 || isNaN(n)) throw TypeError('n must be a positive number');
    this._maxListeners = n;
    return this;
};
EventEmitter.prototype.emit = function(type) {
    var er, handler, len, args, i, listeners;
    if (!this._events) this._events = {};
    // If there is no 'error' event listener then throw.
    if (type === 'error') {
        if (!this._events.error || isObject(this._events.error) && !this._events.error.length) {
            er = arguments[1];
            if (er instanceof Error) throw er; // Unhandled 'error' event
            else {
                // At least give some kind of context to the user
                var err = new Error('Uncaught, unspecified "error" event. (' + er + ')');
                err.context = er;
                throw err;
            }
        }
    }
    handler = this._events[type];
    if (isUndefined(handler)) return false;
    if (isFunction(handler)) switch(arguments.length){
        // fast cases
        case 1:
            handler.call(this);
            break;
        case 2:
            handler.call(this, arguments[1]);
            break;
        case 3:
            handler.call(this, arguments[1], arguments[2]);
            break;
        // slower
        default:
            args = Array.prototype.slice.call(arguments, 1);
            handler.apply(this, args);
    }
    else if (isObject(handler)) {
        args = Array.prototype.slice.call(arguments, 1);
        listeners = handler.slice();
        len = listeners.length;
        for(i = 0; i < len; i++)listeners[i].apply(this, args);
    }
    return true;
};
EventEmitter.prototype.addListener = function(type, listener) {
    var m;
    if (!isFunction(listener)) throw TypeError('listener must be a function');
    if (!this._events) this._events = {};
    // To avoid recursion in the case that type === "newListener"! Before
    // adding it to the listeners, first emit "newListener".
    if (this._events.newListener) this.emit('newListener', type, isFunction(listener.listener) ? listener.listener : listener);
    if (!this._events[type]) // Optimize the case of one listener. Don't need the extra array object.
    this._events[type] = listener;
    else if (isObject(this._events[type])) // If we've already got an array, just append.
    this._events[type].push(listener);
    else // Adding the second element, need to change to array.
    this._events[type] = [
        this._events[type],
        listener
    ];
    // Check for listener leak
    if (isObject(this._events[type]) && !this._events[type].warned) {
        if (!isUndefined(this._maxListeners)) m = this._maxListeners;
        else m = EventEmitter.defaultMaxListeners;
        if (m && m > 0 && this._events[type].length > m) {
            this._events[type].warned = true;
            console.error("(node) warning: possible EventEmitter memory leak detected. %d listeners added. Use emitter.setMaxListeners() to increase limit.", this._events[type].length);
            if (typeof console.trace === 'function') // not supported in IE 10
            console.trace();
        }
    }
    return this;
};
EventEmitter.prototype.on = EventEmitter.prototype.addListener;
EventEmitter.prototype.once = function(type, listener) {
    if (!isFunction(listener)) throw TypeError('listener must be a function');
    var fired = false;
    function g() {
        this.removeListener(type, g);
        if (!fired) {
            fired = true;
            listener.apply(this, arguments);
        }
    }
    g.listener = listener;
    this.on(type, g);
    return this;
};
// emits a 'removeListener' event iff the listener was removed
EventEmitter.prototype.removeListener = function(type, listener) {
    var list, position, length, i;
    if (!isFunction(listener)) throw TypeError('listener must be a function');
    if (!this._events || !this._events[type]) return this;
    list = this._events[type];
    length = list.length;
    position = -1;
    if (list === listener || isFunction(list.listener) && list.listener === listener) {
        delete this._events[type];
        if (this._events.removeListener) this.emit('removeListener', type, listener);
    } else if (isObject(list)) {
        for(i = length; i-- > 0;)if (list[i] === listener || list[i].listener && list[i].listener === listener) {
            position = i;
            break;
        }
        if (position < 0) return this;
        if (list.length === 1) {
            list.length = 0;
            delete this._events[type];
        } else list.splice(position, 1);
        if (this._events.removeListener) this.emit('removeListener', type, listener);
    }
    return this;
};
EventEmitter.prototype.removeAllListeners = function(type) {
    var key, listeners;
    if (!this._events) return this;
    // not listening for removeListener, no need to emit
    if (!this._events.removeListener) {
        if (arguments.length === 0) this._events = {};
        else if (this._events[type]) delete this._events[type];
        return this;
    }
    // emit removeListener for all listeners on all events
    if (arguments.length === 0) {
        for(key in this._events){
            if (key === 'removeListener') continue;
            this.removeAllListeners(key);
        }
        this.removeAllListeners('removeListener');
        this._events = {};
        return this;
    }
    listeners = this._events[type];
    if (isFunction(listeners)) this.removeListener(type, listeners);
    else if (listeners) // LIFO order
    while(listeners.length)this.removeListener(type, listeners[listeners.length - 1]);
    delete this._events[type];
    return this;
};
EventEmitter.prototype.listeners = function(type) {
    var ret;
    if (!this._events || !this._events[type]) ret = [];
    else if (isFunction(this._events[type])) ret = [
        this._events[type]
    ];
    else ret = this._events[type].slice();
    return ret;
};
EventEmitter.prototype.listenerCount = function(type) {
    if (this._events) {
        var evlistener = this._events[type];
        if (isFunction(evlistener)) return 1;
        else if (evlistener) return evlistener.length;
    }
    return 0;
};
EventEmitter.listenerCount = function(emitter, type) {
    return emitter.listenerCount(type);
};
function isFunction(arg) {
    return typeof arg === 'function';
}
function isNumber(arg) {
    return typeof arg === 'number';
}
function isObject(arg) {
    return typeof arg === 'object' && arg !== null;
}
function isUndefined(arg) {
    return arg === void 0;
}

},{}],"a0E30":[function(require,module,exports) {
'use strict';
function inherits(ctor, superCtor) {
    ctor.prototype = Object.create(superCtor.prototype, {
        constructor: {
            value: ctor,
            enumerable: false,
            writable: true,
            configurable: true
        }
    });
}
module.exports = inherits;

},{}],"6rfof":[function(require,module,exports) {
'use strict';
var merge = require('./functions/merge');
var requestBuilder = {
    /**
   * Get all the queries to send to the client, those queries can used directly
   * with the Algolia client.
   * @private
   * @return {object[]} The queries
   */ _getQueries: function getQueries(index, state) {
        var queries = [];
        // One query for the hits
        queries.push({
            indexName: index,
            params: requestBuilder._getHitsSearchParams(state)
        });
        // One for each disjunctive facets
        state.getRefinedDisjunctiveFacets().forEach(function(refinedFacet) {
            queries.push({
                indexName: index,
                params: requestBuilder._getDisjunctiveFacetSearchParams(state, refinedFacet)
            });
        });
        // maybe more to get the root level of hierarchical facets when activated
        state.getRefinedHierarchicalFacets().forEach(function(refinedFacet) {
            var hierarchicalFacet = state.getHierarchicalFacetByName(refinedFacet);
            var currentRefinement = state.getHierarchicalRefinement(refinedFacet);
            // if we are deeper than level 0 (starting from `beer > IPA`)
            // we want to get the root values
            var separator = state._getHierarchicalFacetSeparator(hierarchicalFacet);
            if (currentRefinement.length > 0 && currentRefinement[0].split(separator).length > 1) queries.push({
                indexName: index,
                params: requestBuilder._getDisjunctiveFacetSearchParams(state, refinedFacet, true)
            });
        });
        return queries;
    },
    /**
   * Build search parameters used to fetch hits
   * @private
   * @return {object.<string, any>}
   */ _getHitsSearchParams: function(state) {
        var facets = state.facets.concat(state.disjunctiveFacets).concat(requestBuilder._getHitsHierarchicalFacetsAttributes(state));
        var facetFilters = requestBuilder._getFacetFilters(state);
        var numericFilters = requestBuilder._getNumericFilters(state);
        var tagFilters = requestBuilder._getTagFilters(state);
        var additionalParams = {
            facets: facets.indexOf('*') > -1 ? [
                '*'
            ] : facets,
            tagFilters: tagFilters
        };
        if (facetFilters.length > 0) additionalParams.facetFilters = facetFilters;
        if (numericFilters.length > 0) additionalParams.numericFilters = numericFilters;
        return merge({}, state.getQueryParams(), additionalParams);
    },
    /**
   * Build search parameters used to fetch a disjunctive facet
   * @private
   * @param  {string} facet the associated facet name
   * @param  {boolean} hierarchicalRootLevel ?? FIXME
   * @return {object}
   */ _getDisjunctiveFacetSearchParams: function(state, facet, hierarchicalRootLevel) {
        var facetFilters = requestBuilder._getFacetFilters(state, facet, hierarchicalRootLevel);
        var numericFilters = requestBuilder._getNumericFilters(state, facet);
        var tagFilters = requestBuilder._getTagFilters(state);
        var additionalParams = {
            hitsPerPage: 1,
            page: 0,
            attributesToRetrieve: [],
            attributesToHighlight: [],
            attributesToSnippet: [],
            tagFilters: tagFilters,
            analytics: false,
            clickAnalytics: false
        };
        var hierarchicalFacet = state.getHierarchicalFacetByName(facet);
        if (hierarchicalFacet) additionalParams.facets = requestBuilder._getDisjunctiveHierarchicalFacetAttribute(state, hierarchicalFacet, hierarchicalRootLevel);
        else additionalParams.facets = facet;
        if (numericFilters.length > 0) additionalParams.numericFilters = numericFilters;
        if (facetFilters.length > 0) additionalParams.facetFilters = facetFilters;
        return merge({}, state.getQueryParams(), additionalParams);
    },
    /**
   * Return the numeric filters in an algolia request fashion
   * @private
   * @param {string} [facetName] the name of the attribute for which the filters should be excluded
   * @return {string[]} the numeric filters in the algolia format
   */ _getNumericFilters: function(state, facetName) {
        if (state.numericFilters) return state.numericFilters;
        var numericFilters = [];
        Object.keys(state.numericRefinements).forEach(function(attribute) {
            var operators = state.numericRefinements[attribute] || {};
            Object.keys(operators).forEach(function(operator) {
                var values = operators[operator] || [];
                if (facetName !== attribute) values.forEach(function(value) {
                    if (Array.isArray(value)) {
                        var vs = value.map(function(v) {
                            return attribute + operator + v;
                        });
                        numericFilters.push(vs);
                    } else numericFilters.push(attribute + operator + value);
                });
            });
        });
        return numericFilters;
    },
    /**
   * Return the tags filters depending
   * @private
   * @return {string}
   */ _getTagFilters: function(state) {
        if (state.tagFilters) return state.tagFilters;
        return state.tagRefinements.join(',');
    },
    /**
   * Build facetFilters parameter based on current refinements. The array returned
   * contains strings representing the facet filters in the algolia format.
   * @private
   * @param  {string} [facet] if set, the current disjunctive facet
   * @return {array.<string>}
   */ _getFacetFilters: function(state, facet, hierarchicalRootLevel) {
        var facetFilters = [];
        var facetsRefinements = state.facetsRefinements || {};
        Object.keys(facetsRefinements).forEach(function(facetName) {
            var facetValues = facetsRefinements[facetName] || [];
            facetValues.forEach(function(facetValue) {
                facetFilters.push(facetName + ':' + facetValue);
            });
        });
        var facetsExcludes = state.facetsExcludes || {};
        Object.keys(facetsExcludes).forEach(function(facetName) {
            var facetValues = facetsExcludes[facetName] || [];
            facetValues.forEach(function(facetValue) {
                facetFilters.push(facetName + ':-' + facetValue);
            });
        });
        var disjunctiveFacetsRefinements = state.disjunctiveFacetsRefinements || {};
        Object.keys(disjunctiveFacetsRefinements).forEach(function(facetName) {
            var facetValues = disjunctiveFacetsRefinements[facetName] || [];
            if (facetName === facet || !facetValues || facetValues.length === 0) return;
            var orFilters = [];
            facetValues.forEach(function(facetValue) {
                orFilters.push(facetName + ':' + facetValue);
            });
            facetFilters.push(orFilters);
        });
        var hierarchicalFacetsRefinements = state.hierarchicalFacetsRefinements || {};
        Object.keys(hierarchicalFacetsRefinements).forEach(function(facetName) {
            var facetValues = hierarchicalFacetsRefinements[facetName] || [];
            var facetValue = facetValues[0];
            if (facetValue === undefined) return;
            var hierarchicalFacet = state.getHierarchicalFacetByName(facetName);
            var separator = state._getHierarchicalFacetSeparator(hierarchicalFacet);
            var rootPath = state._getHierarchicalRootPath(hierarchicalFacet);
            var attributeToRefine;
            var attributesIndex;
            // we ask for parent facet values only when the `facet` is the current hierarchical facet
            if (facet === facetName) {
                // if we are at the root level already, no need to ask for facet values, we get them from
                // the hits query
                if (facetValue.indexOf(separator) === -1 || !rootPath && hierarchicalRootLevel === true || rootPath && rootPath.split(separator).length === facetValue.split(separator).length) return;
                if (!rootPath) {
                    attributesIndex = facetValue.split(separator).length - 2;
                    facetValue = facetValue.slice(0, facetValue.lastIndexOf(separator));
                } else {
                    attributesIndex = rootPath.split(separator).length - 1;
                    facetValue = rootPath;
                }
                attributeToRefine = hierarchicalFacet.attributes[attributesIndex];
            } else {
                attributesIndex = facetValue.split(separator).length - 1;
                attributeToRefine = hierarchicalFacet.attributes[attributesIndex];
            }
            if (attributeToRefine) facetFilters.push([
                attributeToRefine + ':' + facetValue
            ]);
        });
        return facetFilters;
    },
    _getHitsHierarchicalFacetsAttributes: function(state) {
        var out = [];
        return state.hierarchicalFacets.reduce(// ask for as much levels as there's hierarchical refinements
        function getHitsAttributesForHierarchicalFacet(allAttributes, hierarchicalFacet) {
            var hierarchicalRefinement = state.getHierarchicalRefinement(hierarchicalFacet.name)[0];
            // if no refinement, ask for root level
            if (!hierarchicalRefinement) {
                allAttributes.push(hierarchicalFacet.attributes[0]);
                return allAttributes;
            }
            var separator = state._getHierarchicalFacetSeparator(hierarchicalFacet);
            var level = hierarchicalRefinement.split(separator).length;
            var newAttributes = hierarchicalFacet.attributes.slice(0, level + 1);
            return allAttributes.concat(newAttributes);
        }, out);
    },
    _getDisjunctiveHierarchicalFacetAttribute: function(state, hierarchicalFacet, rootLevel) {
        var separator = state._getHierarchicalFacetSeparator(hierarchicalFacet);
        if (rootLevel === true) {
            var rootPath = state._getHierarchicalRootPath(hierarchicalFacet);
            var attributeIndex = 0;
            if (rootPath) attributeIndex = rootPath.split(separator).length;
            return [
                hierarchicalFacet.attributes[attributeIndex]
            ];
        }
        var hierarchicalRefinement = state.getHierarchicalRefinement(hierarchicalFacet.name)[0] || '';
        // if refinement is 'beers > IPA > Flying dog',
        // then we want `facets: ['beers > IPA']` as disjunctive facet (parent level values)
        var parentLevel = hierarchicalRefinement.split(separator).length - 1;
        return hierarchicalFacet.attributes.slice(0, parentLevel + 1);
    },
    getSearchForFacetQuery: function(facetName, query, maxFacetHits, state) {
        var stateForSearchForFacetValues = state.isDisjunctiveFacet(facetName) ? state.clearRefinements(facetName) : state;
        var searchForFacetSearchParameters = {
            facetQuery: query,
            facetName: facetName
        };
        if (typeof maxFacetHits === 'number') searchForFacetSearchParameters.maxFacetHits = maxFacetHits;
        return merge({}, requestBuilder._getHitsSearchParams(stateForSearchForFacetValues), searchForFacetSearchParameters);
    }
};
module.exports = requestBuilder;

},{"./functions/merge":"eGyc5"}],"cs17k":[function(require,module,exports) {
'use strict';
module.exports = '3.8.2';

},{}],"1VQLm":[function(require,module,exports) {
// Copyright Joyent, Inc. and other Node contributors.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit
// persons to whom the Software is furnished to do so, subject to the
// following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
// NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
// USE OR OTHER DEALINGS IN THE SOFTWARE.
function EventEmitter() {
    this._events = this._events || {};
    this._maxListeners = this._maxListeners || undefined;
}
module.exports = EventEmitter;
// Backwards-compat with node 0.10.x
EventEmitter.EventEmitter = EventEmitter;
EventEmitter.prototype._events = undefined;
EventEmitter.prototype._maxListeners = undefined;
// By default EventEmitters will print a warning if more than 10 listeners are
// added to it. This is a useful default which helps finding memory leaks.
EventEmitter.defaultMaxListeners = 10;
// Obviously not all Emitters should be limited to 10. This function allows
// that to be increased. Set to zero for unlimited.
EventEmitter.prototype.setMaxListeners = function(n) {
    if (!isNumber(n) || n < 0 || isNaN(n)) throw TypeError('n must be a positive number');
    this._maxListeners = n;
    return this;
};
EventEmitter.prototype.emit = function(type) {
    var er, handler, len, args, i, listeners;
    if (!this._events) this._events = {};
    // If there is no 'error' event listener then throw.
    if (type === 'error') {
        if (!this._events.error || isObject(this._events.error) && !this._events.error.length) {
            er = arguments[1];
            if (er instanceof Error) throw er; // Unhandled 'error' event
            else {
                // At least give some kind of context to the user
                var err = new Error('Uncaught, unspecified "error" event. (' + er + ')');
                err.context = er;
                throw err;
            }
        }
    }
    handler = this._events[type];
    if (isUndefined(handler)) return false;
    if (isFunction(handler)) switch(arguments.length){
        // fast cases
        case 1:
            handler.call(this);
            break;
        case 2:
            handler.call(this, arguments[1]);
            break;
        case 3:
            handler.call(this, arguments[1], arguments[2]);
            break;
        // slower
        default:
            args = Array.prototype.slice.call(arguments, 1);
            handler.apply(this, args);
    }
    else if (isObject(handler)) {
        args = Array.prototype.slice.call(arguments, 1);
        listeners = handler.slice();
        len = listeners.length;
        for(i = 0; i < len; i++)listeners[i].apply(this, args);
    }
    return true;
};
EventEmitter.prototype.addListener = function(type, listener) {
    var m;
    if (!isFunction(listener)) throw TypeError('listener must be a function');
    if (!this._events) this._events = {};
    // To avoid recursion in the case that type === "newListener"! Before
    // adding it to the listeners, first emit "newListener".
    if (this._events.newListener) this.emit('newListener', type, isFunction(listener.listener) ? listener.listener : listener);
    if (!this._events[type]) // Optimize the case of one listener. Don't need the extra array object.
    this._events[type] = listener;
    else if (isObject(this._events[type])) // If we've already got an array, just append.
    this._events[type].push(listener);
    else // Adding the second element, need to change to array.
    this._events[type] = [
        this._events[type],
        listener
    ];
    // Check for listener leak
    if (isObject(this._events[type]) && !this._events[type].warned) {
        if (!isUndefined(this._maxListeners)) m = this._maxListeners;
        else m = EventEmitter.defaultMaxListeners;
        if (m && m > 0 && this._events[type].length > m) {
            this._events[type].warned = true;
            console.error("(node) warning: possible EventEmitter memory leak detected. %d listeners added. Use emitter.setMaxListeners() to increase limit.", this._events[type].length);
            if (typeof console.trace === 'function') // not supported in IE 10
            console.trace();
        }
    }
    return this;
};
EventEmitter.prototype.on = EventEmitter.prototype.addListener;
EventEmitter.prototype.once = function(type, listener) {
    if (!isFunction(listener)) throw TypeError('listener must be a function');
    var fired = false;
    function g() {
        this.removeListener(type, g);
        if (!fired) {
            fired = true;
            listener.apply(this, arguments);
        }
    }
    g.listener = listener;
    this.on(type, g);
    return this;
};
// emits a 'removeListener' event iff the listener was removed
EventEmitter.prototype.removeListener = function(type, listener) {
    var list, position, length, i;
    if (!isFunction(listener)) throw TypeError('listener must be a function');
    if (!this._events || !this._events[type]) return this;
    list = this._events[type];
    length = list.length;
    position = -1;
    if (list === listener || isFunction(list.listener) && list.listener === listener) {
        delete this._events[type];
        if (this._events.removeListener) this.emit('removeListener', type, listener);
    } else if (isObject(list)) {
        for(i = length; i-- > 0;)if (list[i] === listener || list[i].listener && list[i].listener === listener) {
            position = i;
            break;
        }
        if (position < 0) return this;
        if (list.length === 1) {
            list.length = 0;
            delete this._events[type];
        } else list.splice(position, 1);
        if (this._events.removeListener) this.emit('removeListener', type, listener);
    }
    return this;
};
EventEmitter.prototype.removeAllListeners = function(type) {
    var key, listeners;
    if (!this._events) return this;
    // not listening for removeListener, no need to emit
    if (!this._events.removeListener) {
        if (arguments.length === 0) this._events = {};
        else if (this._events[type]) delete this._events[type];
        return this;
    }
    // emit removeListener for all listeners on all events
    if (arguments.length === 0) {
        for(key in this._events){
            if (key === 'removeListener') continue;
            this.removeAllListeners(key);
        }
        this.removeAllListeners('removeListener');
        this._events = {};
        return this;
    }
    listeners = this._events[type];
    if (isFunction(listeners)) this.removeListener(type, listeners);
    else if (listeners) // LIFO order
    while(listeners.length)this.removeListener(type, listeners[listeners.length - 1]);
    delete this._events[type];
    return this;
};
EventEmitter.prototype.listeners = function(type) {
    var ret;
    if (!this._events || !this._events[type]) ret = [];
    else if (isFunction(this._events[type])) ret = [
        this._events[type]
    ];
    else ret = this._events[type].slice();
    return ret;
};
EventEmitter.prototype.listenerCount = function(type) {
    if (this._events) {
        var evlistener = this._events[type];
        if (isFunction(evlistener)) return 1;
        else if (evlistener) return evlistener.length;
    }
    return 0;
};
EventEmitter.listenerCount = function(emitter, type) {
    return emitter.listenerCount(type);
};
function isFunction(arg) {
    return typeof arg === 'function';
}
function isNumber(arg) {
    return typeof arg === 'number';
}
function isObject(arg) {
    return typeof arg === 'object' && arg !== null;
}
function isUndefined(arg) {
    return arg === void 0;
}

},{}],"kdZTz":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "isIndexWidget", ()=>isIndexWidget
);
var _algoliasearchHelper = require("algoliasearch-helper");
var _algoliasearchHelperDefault = parcelHelpers.interopDefault(_algoliasearchHelper);
var _utils = require("../../lib/utils");
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        if (enumerableOnly) symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        });
        keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = arguments[i] != null ? arguments[i] : {};
        if (i % 2) ownKeys(Object(source), true).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        });
        else if (Object.getOwnPropertyDescriptors) Object.defineProperties(target, Object.getOwnPropertyDescriptors(source));
        else ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
function _toConsumableArray(arr) {
    return _arrayWithoutHoles(arr) || _iterableToArray(arr) || _unsupportedIterableToArray(arr) || _nonIterableSpread();
}
function _nonIterableSpread() {
    throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
}
function _unsupportedIterableToArray(o, minLen) {
    if (!o) return;
    if (typeof o === "string") return _arrayLikeToArray(o, minLen);
    var n = Object.prototype.toString.call(o).slice(8, -1);
    if (n === "Object" && o.constructor) n = o.constructor.name;
    if (n === "Map" || n === "Set") return Array.from(o);
    if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen);
}
function _iterableToArray(iter) {
    if (typeof Symbol !== "undefined" && Symbol.iterator in Object(iter)) return Array.from(iter);
}
function _arrayWithoutHoles(arr) {
    if (Array.isArray(arr)) return _arrayLikeToArray(arr);
}
function _arrayLikeToArray(arr, len) {
    if (len == null || len > arr.length) len = arr.length;
    for(var i = 0, arr2 = new Array(len); i < len; i++)arr2[i] = arr[i];
    return arr2;
}
function _objectWithoutProperties(source, excluded) {
    if (source == null) return {};
    var target = _objectWithoutPropertiesLoose(source, excluded);
    var key, i;
    if (Object.getOwnPropertySymbols) {
        var sourceSymbolKeys = Object.getOwnPropertySymbols(source);
        for(i = 0; i < sourceSymbolKeys.length; i++){
            key = sourceSymbolKeys[i];
            if (excluded.indexOf(key) >= 0) continue;
            if (!Object.prototype.propertyIsEnumerable.call(source, key)) continue;
            target[key] = source[key];
        }
    }
    return target;
}
function _objectWithoutPropertiesLoose(source, excluded) {
    if (source == null) return {};
    var target = {};
    var sourceKeys = Object.keys(source);
    var key, i;
    for(i = 0; i < sourceKeys.length; i++){
        key = sourceKeys[i];
        if (excluded.indexOf(key) >= 0) continue;
        target[key] = source[key];
    }
    return target;
}
var withUsage = _utils.createDocumentationMessageGenerator({
    name: 'index-widget'
});
function isIndexWidget(widget) {
    return widget.$$type === 'ais.index';
}
/**
 * This is the same content as helper._change / setState, but allowing for extra
 * UiState to be synchronized.
 * see: https://github.com/algolia/algoliasearch-helper-js/blob/6b835ffd07742f2d6b314022cce6848f5cfecd4a/src/algoliasearch.helper.js#L1311-L1324
 */ function privateHelperSetState(helper, _ref) {
    var state = _ref.state, isPageReset = _ref.isPageReset, _uiState = _ref._uiState;
    if (state !== helper.state) {
        helper.state = state;
        helper.emit('change', {
            state: helper.state,
            results: helper.lastResults,
            isPageReset: isPageReset,
            _uiState: _uiState
        });
    }
}
function getLocalWidgetsUiState(widgets, widgetStateOptions) {
    var initialUiState = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : {};
    return widgets.reduce(function(uiState, widget) {
        if (isIndexWidget(widget)) return uiState;
        if (!widget.getWidgetUiState && !widget.getWidgetState) return uiState;
        if (widget.getWidgetUiState) return widget.getWidgetUiState(uiState, widgetStateOptions);
        return widget.getWidgetState(uiState, widgetStateOptions);
    }, initialUiState);
}
function getLocalWidgetsSearchParameters(widgets, widgetSearchParametersOptions) {
    var initialSearchParameters = widgetSearchParametersOptions.initialSearchParameters, rest = _objectWithoutProperties(widgetSearchParametersOptions, [
        "initialSearchParameters"
    ]);
    return widgets.filter(function(widget) {
        return !isIndexWidget(widget);
    }).reduce(function(state, widget) {
        if (!widget.getWidgetSearchParameters) return state;
        return widget.getWidgetSearchParameters(state, rest);
    }, initialSearchParameters);
}
function resetPageFromWidgets(widgets) {
    var indexWidgets = widgets.filter(isIndexWidget);
    if (indexWidgets.length === 0) return;
    indexWidgets.forEach(function(widget) {
        var widgetHelper = widget.getHelper();
        privateHelperSetState(widgetHelper, {
            state: widgetHelper.state.resetPage(),
            isPageReset: true
        });
        resetPageFromWidgets(widget.getWidgets());
    });
}
function resolveScopedResultsFromWidgets(widgets) {
    var indexWidgets = widgets.filter(isIndexWidget);
    return indexWidgets.reduce(function(scopedResults, current) {
        return scopedResults.concat.apply(scopedResults, [
            {
                indexId: current.getIndexId(),
                results: current.getResults(),
                helper: current.getHelper()
            }
        ].concat(_toConsumableArray(resolveScopedResultsFromWidgets(current.getWidgets()))));
    }, []);
}
var index = function index(widgetParams) {
    if (widgetParams === undefined || widgetParams.indexName === undefined) throw new Error(withUsage('The `indexName` option is required.'));
    var indexName = widgetParams.indexName, _widgetParams$indexId = widgetParams.indexId, indexId = _widgetParams$indexId === void 0 ? indexName : _widgetParams$indexId;
    var localWidgets = [];
    var localUiState = {};
    var localInstantSearchInstance = null;
    var localParent = null;
    var helper = null;
    var derivedHelper = null;
    return {
        $$type: 'ais.index',
        $$widgetType: 'ais.index',
        getIndexName: function getIndexName() {
            return indexName;
        },
        getIndexId: function getIndexId() {
            return indexId;
        },
        getHelper: function getHelper() {
            return helper;
        },
        getResults: function getResults() {
            return derivedHelper && derivedHelper.lastResults;
        },
        getScopedResults: function getScopedResults() {
            var widgetParent = this.getParent(); // If the widget is the root, we consider itself as the only sibling.
            var widgetSiblings = widgetParent ? widgetParent.getWidgets() : [
                this
            ];
            return resolveScopedResultsFromWidgets(widgetSiblings);
        },
        getParent: function getParent() {
            return localParent;
        },
        createURL: function createURL(nextState) {
            return localInstantSearchInstance._createURL(_defineProperty({}, indexId, getLocalWidgetsUiState(localWidgets, {
                searchParameters: nextState,
                helper: helper
            })));
        },
        getWidgets: function getWidgets() {
            return localWidgets;
        },
        addWidgets: function addWidgets(widgets) {
            var _this = this;
            if (!Array.isArray(widgets)) throw new Error(withUsage('The `addWidgets` method expects an array of widgets.'));
            if (widgets.some(function(widget) {
                return typeof widget.init !== 'function' && typeof widget.render !== 'function';
            })) throw new Error(withUsage('The widget definition expects a `render` and/or an `init` method.'));
            localWidgets = localWidgets.concat(widgets);
            if (localInstantSearchInstance && Boolean(widgets.length)) {
                privateHelperSetState(helper, {
                    state: getLocalWidgetsSearchParameters(localWidgets, {
                        uiState: localUiState,
                        initialSearchParameters: helper.state
                    }),
                    _uiState: localUiState
                }); // We compute the render state before calling `init` in a separate loop
                // to construct the whole render state object that is then passed to
                // `init`.
                widgets.forEach(function(widget) {
                    if (widget.getRenderState) {
                        var renderState = widget.getRenderState(localInstantSearchInstance.renderState[_this.getIndexId()] || {}, {
                            uiState: localInstantSearchInstance._initialUiState,
                            helper: _this.getHelper(),
                            parent: _this,
                            instantSearchInstance: localInstantSearchInstance,
                            state: helper.state,
                            renderState: localInstantSearchInstance.renderState,
                            templatesConfig: localInstantSearchInstance.templatesConfig,
                            createURL: _this.createURL,
                            scopedResults: [],
                            searchMetadata: {
                                isSearchStalled: localInstantSearchInstance._isSearchStalled
                            }
                        });
                        storeRenderState({
                            renderState: renderState,
                            instantSearchInstance: localInstantSearchInstance,
                            parent: _this
                        });
                    }
                });
                widgets.forEach(function(widget) {
                    if (widget.init) widget.init({
                        helper: helper,
                        parent: _this,
                        uiState: localInstantSearchInstance._initialUiState,
                        instantSearchInstance: localInstantSearchInstance,
                        state: helper.state,
                        renderState: localInstantSearchInstance.renderState,
                        templatesConfig: localInstantSearchInstance.templatesConfig,
                        createURL: _this.createURL,
                        scopedResults: [],
                        searchMetadata: {
                            isSearchStalled: localInstantSearchInstance._isSearchStalled
                        }
                    });
                });
                localInstantSearchInstance.scheduleSearch();
            }
            return this;
        },
        removeWidgets: function removeWidgets(widgets) {
            var _this2 = this;
            if (!Array.isArray(widgets)) throw new Error(withUsage('The `removeWidgets` method expects an array of widgets.'));
            if (widgets.some(function(widget) {
                return typeof widget.dispose !== 'function';
            })) throw new Error(withUsage('The widget definition expects a `dispose` method.'));
            localWidgets = localWidgets.filter(function(widget) {
                return widgets.indexOf(widget) === -1;
            });
            if (localInstantSearchInstance && Boolean(widgets.length)) {
                var nextState = widgets.reduce(function(state, widget) {
                    // the `dispose` method exists at this point we already assert it
                    var next = widget.dispose({
                        helper: helper,
                        state: state,
                        parent: _this2
                    });
                    return next || state;
                }, helper.state);
                localUiState = getLocalWidgetsUiState(localWidgets, {
                    searchParameters: nextState,
                    helper: helper
                });
                helper.setState(getLocalWidgetsSearchParameters(localWidgets, {
                    uiState: localUiState,
                    initialSearchParameters: nextState
                }));
                if (localWidgets.length) localInstantSearchInstance.scheduleSearch();
            }
            return this;
        },
        init: function init(_ref2) {
            var _this3 = this;
            var instantSearchInstance = _ref2.instantSearchInstance, parent = _ref2.parent, uiState = _ref2.uiState;
            if (helper !== null) // helper is already initialized, therefore we do not need to set up
            // any listeners
            return;
            localInstantSearchInstance = instantSearchInstance;
            localParent = parent;
            localUiState = uiState[indexId] || {}; // The `mainHelper` is already defined at this point. The instance is created
            // inside InstantSearch at the `start` method, which occurs before the `init`
            // step.
            var mainHelper = instantSearchInstance.mainHelper;
            var parameters = getLocalWidgetsSearchParameters(localWidgets, {
                uiState: localUiState,
                initialSearchParameters: new _algoliasearchHelperDefault.default.SearchParameters({
                    index: indexName
                })
            }); // This Helper is only used for state management we do not care about the
            // `searchClient`. Only the "main" Helper created at the `InstantSearch`
            // level is aware of the client.
            helper = _algoliasearchHelperDefault.default({}, parameters.index, parameters); // We forward the call to `search` to the "main" instance of the Helper
            // which is responsible for managing the queries (it's the only one that is
            // aware of the `searchClient`).
            helper.search = function() {
                if (instantSearchInstance.onStateChange) {
                    instantSearchInstance.onStateChange({
                        uiState: instantSearchInstance.mainIndex.getWidgetUiState({}),
                        setUiState: instantSearchInstance.setUiState.bind(instantSearchInstance)
                    }); // We don't trigger a search when controlled because it becomes the
                    // responsibility of `setUiState`.
                    return mainHelper;
                }
                return mainHelper.search();
            };
            helper.searchWithoutTriggeringOnStateChange = function() {
                return mainHelper.search();
            }; // We use the same pattern for the `searchForFacetValues`.
            helper.searchForFacetValues = function(facetName, facetValue, maxFacetHits, userState) {
                var state = helper.state.setQueryParameters(userState);
                return mainHelper.searchForFacetValues(facetName, facetValue, maxFacetHits, state);
            };
            derivedHelper = mainHelper.derive(function() {
                return _utils.mergeSearchParameters.apply(void 0, _toConsumableArray(_utils.resolveSearchParameters(_this3)));
            }); // Subscribe to the Helper state changes for the page before widgets
            // are initialized. This behavior mimics the original one of the Helper.
            // It makes sense to replicate it at the `init` step. We have another
            // listener on `change` below, once `init` is done.
            helper.on('change', function(_ref3) {
                var isPageReset = _ref3.isPageReset;
                if (isPageReset) resetPageFromWidgets(localWidgets);
            });
            derivedHelper.on('search', function() {
                // The index does not manage the "staleness" of the search. This is the
                // responsibility of the main instance. It does not make sense to manage
                // it at the index level because it's either: all of them or none of them
                // that are stalled. The queries are performed into a single network request.
                instantSearchInstance.scheduleStalledRender();
                _utils.checkIndexUiState({
                    index: _this3,
                    indexUiState: localUiState
                });
            });
            derivedHelper.on('result', function(_ref4) {
                var results = _ref4.results;
                // The index does not render the results it schedules a new render
                // to let all the other indices emit their own results. It allows us to
                // run the render process in one pass.
                instantSearchInstance.scheduleRender(); // the derived helper is the one which actually searches, but the helper
                // which is exposed e.g. via instance.helper, doesn't search, and thus
                // does not have access to lastResults, which it used to in pre-federated
                // search behavior.
                helper.lastResults = results;
            }); // We compute the render state before calling `init` in a separate loop
            // to construct the whole render state object that is then passed to
            // `init`.
            localWidgets.forEach(function(widget) {
                if (widget.getRenderState) {
                    var renderState = widget.getRenderState(instantSearchInstance.renderState[_this3.getIndexId()] || {}, {
                        uiState: uiState,
                        helper: helper,
                        parent: _this3,
                        instantSearchInstance: instantSearchInstance,
                        state: helper.state,
                        renderState: instantSearchInstance.renderState,
                        templatesConfig: instantSearchInstance.templatesConfig,
                        createURL: _this3.createURL,
                        scopedResults: [],
                        searchMetadata: {
                            isSearchStalled: instantSearchInstance._isSearchStalled
                        }
                    });
                    storeRenderState({
                        renderState: renderState,
                        instantSearchInstance: instantSearchInstance,
                        parent: _this3
                    });
                }
            });
            localWidgets.forEach(function(widget) {
                _utils.warning(// aka we warn if there's _only_ getWidgetState
                !widget.getWidgetState || Boolean(widget.getWidgetUiState), 'The `getWidgetState` method is renamed `getWidgetUiState` and will no longer exist under that name in InstantSearch.js 5.x. Please use `getWidgetUiState` instead.');
                if (widget.init) widget.init({
                    uiState: uiState,
                    helper: helper,
                    parent: _this3,
                    instantSearchInstance: instantSearchInstance,
                    state: helper.state,
                    renderState: instantSearchInstance.renderState,
                    templatesConfig: instantSearchInstance.templatesConfig,
                    createURL: _this3.createURL,
                    scopedResults: [],
                    searchMetadata: {
                        isSearchStalled: instantSearchInstance._isSearchStalled
                    }
                });
            }); // Subscribe to the Helper state changes for the `uiState` once widgets
            // are initialized. Until the first render, state changes are part of the
            // configuration step. This is mainly for backward compatibility with custom
            // widgets. When the subscription happens before the `init` step, the (static)
            // configuration of the widget is pushed in the URL. That's what we want to avoid.
            // https://github.com/algolia/instantsearch.js/pull/994/commits/4a672ae3fd78809e213de0368549ef12e9dc9454
            helper.on('change', function(event) {
                var state = event.state;
                var _uiState = event._uiState;
                localUiState = getLocalWidgetsUiState(localWidgets, {
                    searchParameters: state,
                    helper: helper
                }, _uiState || {}); // We don't trigger an internal change when controlled because it
                // becomes the responsibility of `setUiState`.
                if (!instantSearchInstance.onStateChange) instantSearchInstance.onInternalStateChange();
            });
        },
        render: function render(_ref5) {
            var _this4 = this;
            var instantSearchInstance = _ref5.instantSearchInstance;
            if (!this.getResults()) return;
            localWidgets.forEach(function(widget) {
                if (widget.getRenderState) {
                    var renderState = widget.getRenderState(instantSearchInstance.renderState[_this4.getIndexId()] || {}, {
                        helper: _this4.getHelper(),
                        parent: _this4,
                        instantSearchInstance: instantSearchInstance,
                        results: _this4.getResults(),
                        scopedResults: _this4.getScopedResults(),
                        state: _this4.getResults()._state,
                        renderState: instantSearchInstance.renderState,
                        templatesConfig: instantSearchInstance.templatesConfig,
                        createURL: _this4.createURL,
                        searchMetadata: {
                            isSearchStalled: instantSearchInstance._isSearchStalled
                        }
                    });
                    storeRenderState({
                        renderState: renderState,
                        instantSearchInstance: instantSearchInstance,
                        parent: _this4
                    });
                }
            });
            localWidgets.forEach(function(widget) {
                // At this point, all the variables used below are set. Both `helper`
                // and `derivedHelper` have been created at the `init` step. The attribute
                // `lastResults` might be `null` though. It's possible that a stalled render
                // happens before the result e.g with a dynamically added index the request might
                // be delayed. The render is triggered for the complete tree but some parts do
                // not have results yet.
                if (widget.render) widget.render({
                    helper: helper,
                    parent: _this4,
                    instantSearchInstance: instantSearchInstance,
                    results: _this4.getResults(),
                    scopedResults: _this4.getScopedResults(),
                    state: _this4.getResults()._state,
                    renderState: instantSearchInstance.renderState,
                    templatesConfig: instantSearchInstance.templatesConfig,
                    createURL: _this4.createURL,
                    searchMetadata: {
                        isSearchStalled: instantSearchInstance._isSearchStalled
                    }
                });
            });
        },
        dispose: function dispose() {
            var _this5 = this;
            localWidgets.forEach(function(widget) {
                if (widget.dispose) // The dispose function is always called once the instance is started
                // (it's an effect of `removeWidgets`). The index is initialized and
                // the Helper is available. We don't care about the return value of
                // `dispose` because the index is removed. We can't call `removeWidgets`
                // because we want to keep the widgets on the instance, to allow idempotent
                // operations on `add` & `remove`.
                widget.dispose({
                    helper: helper,
                    state: helper.state,
                    parent: _this5
                });
            });
            localInstantSearchInstance = null;
            localParent = null;
            helper.removeAllListeners();
            helper = null;
            derivedHelper.detach();
            derivedHelper = null;
        },
        getWidgetUiState: function getWidgetUiState(uiState) {
            return localWidgets.filter(isIndexWidget).reduce(function(previousUiState, innerIndex) {
                return innerIndex.getWidgetUiState(previousUiState);
            }, _objectSpread(_objectSpread({}, uiState), {}, _defineProperty({}, this.getIndexId(), localUiState)));
        },
        getWidgetState: function getWidgetState(uiState) {
            _utils.warning(false, 'The `getWidgetState` method is renamed `getWidgetUiState` and will no longer exist under that name in InstantSearch.js 5.x. Please use `getWidgetUiState` instead.');
            return this.getWidgetUiState(uiState);
        },
        getWidgetSearchParameters: function getWidgetSearchParameters(searchParameters, _ref6) {
            var uiState = _ref6.uiState;
            return getLocalWidgetsSearchParameters(localWidgets, {
                uiState: uiState,
                initialSearchParameters: searchParameters
            });
        },
        refreshUiState: function refreshUiState() {
            localUiState = getLocalWidgetsUiState(localWidgets, {
                searchParameters: this.getHelper().state,
                helper: this.getHelper()
            });
        }
    };
};
exports.default = index;
function storeRenderState(_ref7) {
    var renderState = _ref7.renderState, instantSearchInstance = _ref7.instantSearchInstance, parent = _ref7.parent;
    var parentIndexName = parent ? parent.getIndexId() : instantSearchInstance.mainIndex.getIndexId();
    instantSearchInstance.renderState = _objectSpread(_objectSpread({}, instantSearchInstance.renderState), {}, _defineProperty({}, parentIndexName, _objectSpread(_objectSpread({}, instantSearchInstance.renderState[parentIndexName]), renderState)));
}

},{"algoliasearch-helper":"jGqjt","../../lib/utils":"etVYs","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"etVYs":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "capitalize", ()=>_capitalizeDefault.default
);
parcelHelpers.export(exports, "defer", ()=>_deferDefault.default
);
parcelHelpers.export(exports, "isDomElement", ()=>_isDomElementDefault.default
);
parcelHelpers.export(exports, "getContainerNode", ()=>_getContainerNodeDefault.default
);
parcelHelpers.export(exports, "isSpecialClick", ()=>_isSpecialClickDefault.default
);
parcelHelpers.export(exports, "prepareTemplateProps", ()=>_prepareTemplatePropsDefault.default
);
parcelHelpers.export(exports, "renderTemplate", ()=>_renderTemplateDefault.default
);
parcelHelpers.export(exports, "getRefinements", ()=>_getRefinementsDefault.default
);
parcelHelpers.export(exports, "clearRefinements", ()=>_clearRefinementsDefault.default
);
parcelHelpers.export(exports, "escapeRefinement", ()=>_escapeRefinementDefault.default
);
parcelHelpers.export(exports, "unescapeRefinement", ()=>_unescapeRefinementDefault.default
);
parcelHelpers.export(exports, "checkRendering", ()=>_checkRenderingDefault.default
);
parcelHelpers.export(exports, "checkIndexUiState", ()=>_checkIndexUiState.checkIndexUiState
);
parcelHelpers.export(exports, "getPropertyByPath", ()=>_getPropertyByPathDefault.default
);
parcelHelpers.export(exports, "getObjectType", ()=>_getObjectTypeDefault.default
);
parcelHelpers.export(exports, "noop", ()=>_noopDefault.default
);
parcelHelpers.export(exports, "isFiniteNumber", ()=>_isFiniteNumberDefault.default
);
parcelHelpers.export(exports, "isPlainObject", ()=>_isPlainObjectDefault.default
);
parcelHelpers.export(exports, "uniq", ()=>_uniqDefault.default
);
parcelHelpers.export(exports, "range", ()=>_rangeDefault.default
);
parcelHelpers.export(exports, "isEqual", ()=>_isEqualDefault.default
);
parcelHelpers.export(exports, "escape", ()=>_escapeDefault.default
);
parcelHelpers.export(exports, "unescape", ()=>_unescapeDefault.default
);
parcelHelpers.export(exports, "concatHighlightedParts", ()=>_concatHighlightedPartsDefault.default
);
parcelHelpers.export(exports, "getHighlightedParts", ()=>_getHighlightedPartsDefault.default
);
parcelHelpers.export(exports, "getHighlightFromSiblings", ()=>_getHighlightFromSiblingsDefault.default
);
parcelHelpers.export(exports, "reverseHighlightedParts", ()=>_reverseHighlightedPartsDefault.default
);
parcelHelpers.export(exports, "find", ()=>_findDefault.default
);
parcelHelpers.export(exports, "findIndex", ()=>_findIndexDefault.default
);
parcelHelpers.export(exports, "mergeSearchParameters", ()=>_mergeSearchParametersDefault.default
);
parcelHelpers.export(exports, "resolveSearchParameters", ()=>_resolveSearchParametersDefault.default
);
parcelHelpers.export(exports, "toArray", ()=>_toArrayDefault.default
);
parcelHelpers.export(exports, "warning", ()=>_logger.warning
);
parcelHelpers.export(exports, "deprecate", ()=>_logger.deprecate
);
parcelHelpers.export(exports, "escapeHits", ()=>_escapeHighlight.escapeHits
);
parcelHelpers.export(exports, "TAG_PLACEHOLDER", ()=>_escapeHighlight.TAG_PLACEHOLDER
);
parcelHelpers.export(exports, "TAG_REPLACEMENT", ()=>_escapeHighlight.TAG_REPLACEMENT
);
parcelHelpers.export(exports, "escapeFacets", ()=>_escapeHighlight.escapeFacets
);
parcelHelpers.export(exports, "createDocumentationLink", ()=>_documentation.createDocumentationLink
);
parcelHelpers.export(exports, "createDocumentationMessageGenerator", ()=>_documentation.createDocumentationMessageGenerator
);
parcelHelpers.export(exports, "aroundLatLngToPosition", ()=>_geoSearch.aroundLatLngToPosition
);
parcelHelpers.export(exports, "insideBoundingBoxToBoundingBox", ()=>_geoSearch.insideBoundingBoxToBoundingBox
);
parcelHelpers.export(exports, "addAbsolutePosition", ()=>_hitsAbsolutePosition.addAbsolutePosition
);
parcelHelpers.export(exports, "addQueryID", ()=>_hitsQueryId.addQueryID
);
parcelHelpers.export(exports, "isFacetRefined", ()=>_isFacetRefinedDefault.default
);
parcelHelpers.export(exports, "getAppIdAndApiKey", ()=>_getAppIdAndApiKey.getAppIdAndApiKey
);
parcelHelpers.export(exports, "convertNumericRefinementsToFilters", ()=>_convertNumericRefinementsToFilters.convertNumericRefinementsToFilters
);
parcelHelpers.export(exports, "createConcurrentSafePromise", ()=>_createConcurrentSafePromise.createConcurrentSafePromise
);
parcelHelpers.export(exports, "debounce", ()=>_debounce.debounce
);
parcelHelpers.export(exports, "serializePayload", ()=>_serializer.serializePayload
);
parcelHelpers.export(exports, "deserializePayload", ()=>_serializer.deserializePayload
);
parcelHelpers.export(exports, "getWidgetAttribute", ()=>_getWidgetAttribute.getWidgetAttribute
);
var _capitalize = require("./capitalize");
var _capitalizeDefault = parcelHelpers.interopDefault(_capitalize);
var _defer = require("./defer");
var _deferDefault = parcelHelpers.interopDefault(_defer);
var _isDomElement = require("./isDomElement");
var _isDomElementDefault = parcelHelpers.interopDefault(_isDomElement);
var _getContainerNode = require("./getContainerNode");
var _getContainerNodeDefault = parcelHelpers.interopDefault(_getContainerNode);
var _isSpecialClick = require("./isSpecialClick");
var _isSpecialClickDefault = parcelHelpers.interopDefault(_isSpecialClick);
var _prepareTemplateProps = require("./prepareTemplateProps");
var _prepareTemplatePropsDefault = parcelHelpers.interopDefault(_prepareTemplateProps);
var _renderTemplate = require("./renderTemplate");
var _renderTemplateDefault = parcelHelpers.interopDefault(_renderTemplate);
var _getRefinements = require("./getRefinements");
var _getRefinementsDefault = parcelHelpers.interopDefault(_getRefinements);
var _clearRefinements = require("./clearRefinements");
var _clearRefinementsDefault = parcelHelpers.interopDefault(_clearRefinements);
var _escapeRefinement = require("./escapeRefinement");
var _escapeRefinementDefault = parcelHelpers.interopDefault(_escapeRefinement);
var _unescapeRefinement = require("./unescapeRefinement");
var _unescapeRefinementDefault = parcelHelpers.interopDefault(_unescapeRefinement);
var _checkRendering = require("./checkRendering");
var _checkRenderingDefault = parcelHelpers.interopDefault(_checkRendering);
var _checkIndexUiState = require("./checkIndexUiState");
var _getPropertyByPath = require("./getPropertyByPath");
var _getPropertyByPathDefault = parcelHelpers.interopDefault(_getPropertyByPath);
var _getObjectType = require("./getObjectType");
var _getObjectTypeDefault = parcelHelpers.interopDefault(_getObjectType);
var _noop = require("./noop");
var _noopDefault = parcelHelpers.interopDefault(_noop);
var _isFiniteNumber = require("./isFiniteNumber");
var _isFiniteNumberDefault = parcelHelpers.interopDefault(_isFiniteNumber);
var _isPlainObject = require("./isPlainObject");
var _isPlainObjectDefault = parcelHelpers.interopDefault(_isPlainObject);
var _uniq = require("./uniq");
var _uniqDefault = parcelHelpers.interopDefault(_uniq);
var _range = require("./range");
var _rangeDefault = parcelHelpers.interopDefault(_range);
var _isEqual = require("./isEqual");
var _isEqualDefault = parcelHelpers.interopDefault(_isEqual);
var _escape = require("./escape");
var _escapeDefault = parcelHelpers.interopDefault(_escape);
var _unescape = require("./unescape");
var _unescapeDefault = parcelHelpers.interopDefault(_unescape);
var _concatHighlightedParts = require("./concatHighlightedParts");
var _concatHighlightedPartsDefault = parcelHelpers.interopDefault(_concatHighlightedParts);
var _getHighlightedParts = require("./getHighlightedParts");
var _getHighlightedPartsDefault = parcelHelpers.interopDefault(_getHighlightedParts);
var _getHighlightFromSiblings = require("./getHighlightFromSiblings");
var _getHighlightFromSiblingsDefault = parcelHelpers.interopDefault(_getHighlightFromSiblings);
var _reverseHighlightedParts = require("./reverseHighlightedParts");
var _reverseHighlightedPartsDefault = parcelHelpers.interopDefault(_reverseHighlightedParts);
var _find = require("./find");
var _findDefault = parcelHelpers.interopDefault(_find);
var _findIndex = require("./findIndex");
var _findIndexDefault = parcelHelpers.interopDefault(_findIndex);
var _mergeSearchParameters = require("./mergeSearchParameters");
var _mergeSearchParametersDefault = parcelHelpers.interopDefault(_mergeSearchParameters);
var _resolveSearchParameters = require("./resolveSearchParameters");
var _resolveSearchParametersDefault = parcelHelpers.interopDefault(_resolveSearchParameters);
var _toArray = require("./toArray");
var _toArrayDefault = parcelHelpers.interopDefault(_toArray);
var _logger = require("./logger");
var _escapeHighlight = require("./escape-highlight");
var _documentation = require("./documentation");
var _geoSearch = require("./geo-search");
var _hitsAbsolutePosition = require("./hits-absolute-position");
var _hitsQueryId = require("./hits-query-id");
var _isFacetRefined = require("./isFacetRefined");
var _isFacetRefinedDefault = parcelHelpers.interopDefault(_isFacetRefined);
var _createSendEventForFacet = require("./createSendEventForFacet");
parcelHelpers.exportAll(_createSendEventForFacet, exports);
var _createSendEventForHits = require("./createSendEventForHits");
parcelHelpers.exportAll(_createSendEventForHits, exports);
var _getAppIdAndApiKey = require("./getAppIdAndApiKey");
var _convertNumericRefinementsToFilters = require("./convertNumericRefinementsToFilters");
var _createConcurrentSafePromise = require("./createConcurrentSafePromise");
var _debounce = require("./debounce");
var _serializer = require("./serializer");
var _getWidgetAttribute = require("./getWidgetAttribute");

},{"./capitalize":"1J2wi","./defer":"bO5Os","./isDomElement":"3TY64","./getContainerNode":"ayQ3q","./isSpecialClick":"lKDby","./prepareTemplateProps":"3Knzg","./renderTemplate":"cpZ6z","./getRefinements":false,"./clearRefinements":false,"./escapeRefinement":false,"./unescapeRefinement":false,"./checkRendering":"jF2C6","./checkIndexUiState":"bH0Ll","./getPropertyByPath":"2Q0lT","./getObjectType":"3XQ8P","./noop":"6iazv","./isFiniteNumber":"gQhvL","./isPlainObject":"cIivc","./uniq":"2Q0ce","./range":"1dHGc","./isEqual":"14V8N","./escape":"eLn1u","./unescape":"cWsJ0","./concatHighlightedParts":"12qh7","./getHighlightedParts":"7jCC9","./getHighlightFromSiblings":"fAPc5","./reverseHighlightedParts":"kBcoN","./find":"6Dhef","./findIndex":"8tlAy","./mergeSearchParameters":"9Li6L","./resolveSearchParameters":"a7lVI","./toArray":false,"./logger":"glTTt","./escape-highlight":"59mW0","./documentation":"gLqHy","./geo-search":false,"./hits-absolute-position":"dMQpP","./hits-query-id":"iBpEo","./isFacetRefined":"b5SV4","./createSendEventForFacet":"05go2","./createSendEventForHits":"24sIF","./getAppIdAndApiKey":false,"./convertNumericRefinementsToFilters":"ekxD4","./createConcurrentSafePromise":false,"./debounce":false,"./serializer":"jg61H","./getWidgetAttribute":false,"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"1J2wi":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
function capitalize(text) {
    return text.toString().charAt(0).toUpperCase() + text.toString().slice(1);
}
exports.default = capitalize;

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"gkKU3":[function(require,module,exports) {
exports.interopDefault = function(a) {
    return a && a.__esModule ? a : {
        default: a
    };
};
exports.defineInteropFlag = function(a) {
    Object.defineProperty(a, '__esModule', {
        value: true
    });
};
exports.exportAll = function(source, dest) {
    Object.keys(source).forEach(function(key) {
        if (key === 'default' || key === '__esModule' || dest.hasOwnProperty(key)) return;
        Object.defineProperty(dest, key, {
            enumerable: true,
            get: function() {
                return source[key];
            }
        });
    });
    return dest;
};
exports.export = function(dest, destName, get) {
    Object.defineProperty(dest, destName, {
        enumerable: true,
        get: get
    });
};

},{}],"bO5Os":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var nextMicroTask = Promise.resolve();
var defer = function defer(callback) {
    var progress = null;
    var cancelled = false;
    var fn = function fn() {
        for(var _len = arguments.length, args = new Array(_len), _key = 0; _key < _len; _key++)args[_key] = arguments[_key];
        if (progress !== null) return;
        progress = nextMicroTask.then(function() {
            progress = null;
            if (cancelled) {
                cancelled = false;
                return;
            }
            callback.apply(void 0, args);
        });
    };
    fn.wait = function() {
        if (progress === null) throw new Error('The deferred function should be called before calling `wait()`');
        return progress;
    };
    fn.cancel = function() {
        if (progress === null) return;
        cancelled = true;
    };
    return fn;
};
exports.default = defer;

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"3TY64":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
function isDomElement(object) {
    return object instanceof HTMLElement || Boolean(object) && object.nodeType > 0;
}
exports.default = isDomElement;

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"ayQ3q":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _isDomElement = require("./isDomElement");
var _isDomElementDefault = parcelHelpers.interopDefault(_isDomElement);
/**
 * Return the container. If it's a string, it is considered a
 * css selector and retrieves the first matching element. Otherwise
 * test if it validates that it's a correct DOMElement.
 *
 * @param {string|HTMLElement} selectorOrHTMLElement CSS Selector or container node.
 * @return {HTMLElement} Container node
 * @throws Error when the type is not correct
 */ function getContainerNode(selectorOrHTMLElement) {
    var isSelectorString = typeof selectorOrHTMLElement === 'string';
    var domElement = isSelectorString ? document.querySelector(selectorOrHTMLElement) : selectorOrHTMLElement;
    if (!_isDomElementDefault.default(domElement)) {
        var errorMessage = 'Container must be `string` or `HTMLElement`.';
        if (isSelectorString) errorMessage += " Unable to find ".concat(selectorOrHTMLElement);
        throw new Error(errorMessage);
    }
    return domElement;
}
exports.default = getContainerNode;

},{"./isDomElement":"3TY64","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"lKDby":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
function isSpecialClick(event) {
    var isMiddleClick = event.button === 1;
    return isMiddleClick || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey;
}
exports.default = isSpecialClick;

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"3Knzg":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _uniq = require("./uniq");
var _uniqDefault = parcelHelpers.interopDefault(_uniq);
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        if (enumerableOnly) symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        });
        keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = arguments[i] != null ? arguments[i] : {};
        if (i % 2) ownKeys(Object(source), true).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        });
        else if (Object.getOwnPropertyDescriptors) Object.defineProperties(target, Object.getOwnPropertyDescriptors(source));
        else ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
function _toConsumableArray(arr) {
    return _arrayWithoutHoles(arr) || _iterableToArray(arr) || _unsupportedIterableToArray(arr) || _nonIterableSpread();
}
function _nonIterableSpread() {
    throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
}
function _unsupportedIterableToArray(o, minLen) {
    if (!o) return;
    if (typeof o === "string") return _arrayLikeToArray(o, minLen);
    var n = Object.prototype.toString.call(o).slice(8, -1);
    if (n === "Object" && o.constructor) n = o.constructor.name;
    if (n === "Map" || n === "Set") return Array.from(o);
    if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen);
}
function _iterableToArray(iter) {
    if (typeof Symbol !== "undefined" && Symbol.iterator in Object(iter)) return Array.from(iter);
}
function _arrayWithoutHoles(arr) {
    if (Array.isArray(arr)) return _arrayLikeToArray(arr);
}
function _arrayLikeToArray(arr, len) {
    if (len == null || len > arr.length) len = arr.length;
    for(var i = 0, arr2 = new Array(len); i < len; i++)arr2[i] = arr[i];
    return arr2;
}
function prepareTemplates(defaultTemplates) {
    var templates = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {};
    var allKeys = _uniqDefault.default([].concat(_toConsumableArray(Object.keys(defaultTemplates || {})), _toConsumableArray(Object.keys(templates))));
    return allKeys.reduce(function(config, key) {
        var defaultTemplate = defaultTemplates ? defaultTemplates[key] : undefined;
        var customTemplate = templates[key];
        var isCustomTemplate = customTemplate !== undefined && customTemplate !== defaultTemplate;
        config.templates[key] = isCustomTemplate ? customTemplate // typescript doesn't recognize that this condition asserts customTemplate is defined
         : defaultTemplate;
        config.useCustomCompileOptions[key] = isCustomTemplate;
        return config;
    }, {
        // eslint-disable-next-line @typescript-eslint/consistent-type-assertions
        templates: {},
        // eslint-disable-next-line @typescript-eslint/consistent-type-assertions
        useCustomCompileOptions: {}
    });
}
/**
 * Prepares an object to be passed to the Template widget
 */ function prepareTemplateProps(_ref) {
    var defaultTemplates = _ref.defaultTemplates, templates = _ref.templates, templatesConfig = _ref.templatesConfig;
    var preparedTemplates = prepareTemplates(defaultTemplates, templates);
    return _objectSpread({
        templatesConfig: templatesConfig
    }, preparedTemplates);
}
exports.default = prepareTemplateProps;

},{"./uniq":"2Q0ce","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"2Q0ce":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
function uniq(array) {
    return array.filter(function(value, index, self) {
        return self.indexOf(value) === index;
    });
}
exports.default = uniq;

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"cpZ6z":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _hoganJs = require("hogan.js"); // We add all our template helper methods to the template as lambdas. Note
var _hoganJsDefault = parcelHelpers.interopDefault(_hoganJs);
function _typeof(obj1) {
    if (typeof Symbol === "function" && typeof Symbol.iterator === "symbol") _typeof = function _typeof(obj) {
        return typeof obj;
    };
    else _typeof = function _typeof(obj) {
        return obj && typeof Symbol === "function" && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj;
    };
    return _typeof(obj1);
}
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        if (enumerableOnly) symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        });
        keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = arguments[i] != null ? arguments[i] : {};
        if (i % 2) ownKeys(Object(source), true).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        });
        else if (Object.getOwnPropertyDescriptors) Object.defineProperties(target, Object.getOwnPropertyDescriptors(source));
        else ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
// that lambdas in Mustache are supposed to accept a second argument of
// `render` to get the rendered value, not the literal `{{value}}`. But
// this is currently broken (see https://github.com/twitter/hogan.js/issues/222).
function transformHelpersToHogan() {
    var helpers = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : {};
    var compileOptions = arguments.length > 1 ? arguments[1] : undefined;
    var data = arguments.length > 2 ? arguments[2] : undefined;
    return Object.keys(helpers).reduce(function(acc, helperKey) {
        return _objectSpread(_objectSpread({}, acc), {}, _defineProperty({}, helperKey, function() {
            var _this = this;
            return function(text) {
                var render = function render(value) {
                    return _hoganJsDefault.default.compile(value, compileOptions).render(_this);
                };
                return helpers[helperKey].call(data, text, render);
            };
        }));
    }, {});
}
function renderTemplate(_ref) {
    var templates = _ref.templates, templateKey = _ref.templateKey, compileOptions = _ref.compileOptions, helpers = _ref.helpers, data = _ref.data, bindEvent = _ref.bindEvent;
    var template = templates[templateKey];
    var templateType = _typeof(template);
    var isTemplateString = templateType === 'string';
    var isTemplateFunction = templateType === 'function';
    if (!isTemplateString && !isTemplateFunction) throw new Error("Template must be 'string' or 'function', was '".concat(templateType, "' (key: ").concat(templateKey, ")"));
    if (isTemplateFunction) return template(data, bindEvent);
    var transformedHelpers = transformHelpersToHogan(helpers, compileOptions, data);
    return _hoganJsDefault.default.compile(template, compileOptions).render(_objectSpread(_objectSpread({}, data), {}, {
        helpers: transformedHelpers
    })).replace(/[ \n\r\t\f\xA0]+/g, function(spaces) {
        return spaces.replace(/(^|\xA0+)[^\xA0]+/g, '$1 ');
    }).trim();
}
exports.default = renderTemplate;

},{"hogan.js":"gkYEi","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"gkYEi":[function(require,module,exports) {
/*
 *  Copyright 2011 Twitter, Inc.
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */ // This file is for use with Node.js. See dist/ for browser files.
var Hogan = require('./compiler');
Hogan.Template = require('./template').Template;
Hogan.template = Hogan.Template;
module.exports = Hogan;

},{"./compiler":"ezTiX","./template":"kCFri"}],"ezTiX":[function(require,module,exports) {
/*
 *  Copyright 2011 Twitter, Inc.
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */ (function(Hogan) {
    // Setup regex  assignments
    // remove whitespace according to Mustache spec
    var rIsWhitespace = /\S/, rQuot = /\"/g, rNewline = /\n/g, rCr = /\r/g, rSlash = /\\/g, rLineSep = /\u2028/, rParagraphSep = /\u2029/;
    Hogan.tags = {
        '#': 1,
        '^': 2,
        '<': 3,
        '$': 4,
        '/': 5,
        '!': 6,
        '>': 7,
        '=': 8,
        '_v': 9,
        '{': 10,
        '&': 11,
        '_t': 12
    };
    Hogan.scan = function scan(text1, delimiters1) {
        var len = text1.length, IN_TEXT = 0, IN_TAG_TYPE = 1, IN_TAG = 2, state = IN_TEXT, tagType = null, tag = null, buf = '', tokens = [], seenTag = false, i = 0, lineStart = 0, otag = '{{', ctag = '}}';
        function addBuf() {
            if (buf.length > 0) {
                tokens.push({
                    tag: '_t',
                    text: new String(buf)
                });
                buf = '';
            }
        }
        function lineIsWhitespace() {
            var isAllWhitespace = true;
            for(var j = lineStart; j < tokens.length; j++){
                isAllWhitespace = Hogan.tags[tokens[j].tag] < Hogan.tags['_v'] || tokens[j].tag == '_t' && tokens[j].text.match(rIsWhitespace) === null;
                if (!isAllWhitespace) return false;
            }
            return isAllWhitespace;
        }
        function filterLine(haveSeenTag, noNewLine) {
            addBuf();
            if (haveSeenTag && lineIsWhitespace()) {
                for(var j = lineStart, next; j < tokens.length; j++)if (tokens[j].text) {
                    if ((next = tokens[j + 1]) && next.tag == '>') // set indent to token value
                    next.indent = tokens[j].text.toString();
                    tokens.splice(j, 1);
                }
            } else if (!noNewLine) tokens.push({
                tag: '\n'
            });
            seenTag = false;
            lineStart = tokens.length;
        }
        function changeDelimiters(text, index) {
            var close = '=' + ctag, closeIndex = text.indexOf(close, index), delimiters = trim(text.substring(text.indexOf('=', index) + 1, closeIndex)).split(' ');
            otag = delimiters[0];
            ctag = delimiters[delimiters.length - 1];
            return closeIndex + close.length - 1;
        }
        if (delimiters1) {
            delimiters1 = delimiters1.split(' ');
            otag = delimiters1[0];
            ctag = delimiters1[1];
        }
        for(i = 0; i < len; i++){
            if (state == IN_TEXT) {
                if (tagChange(otag, text1, i)) {
                    --i;
                    addBuf();
                    state = IN_TAG_TYPE;
                } else if (text1.charAt(i) == '\n') filterLine(seenTag);
                else buf += text1.charAt(i);
            } else if (state == IN_TAG_TYPE) {
                i += otag.length - 1;
                tag = Hogan.tags[text1.charAt(i + 1)];
                tagType = tag ? text1.charAt(i + 1) : '_v';
                if (tagType == '=') {
                    i = changeDelimiters(text1, i);
                    state = IN_TEXT;
                } else {
                    if (tag) i++;
                    state = IN_TAG;
                }
                seenTag = i;
            } else if (tagChange(ctag, text1, i)) {
                tokens.push({
                    tag: tagType,
                    n: trim(buf),
                    otag: otag,
                    ctag: ctag,
                    i: tagType == '/' ? seenTag - otag.length : i + ctag.length
                });
                buf = '';
                i += ctag.length - 1;
                state = IN_TEXT;
                if (tagType == '{') {
                    if (ctag == '}}') i++;
                    else cleanTripleStache(tokens[tokens.length - 1]);
                }
            } else buf += text1.charAt(i);
        }
        filterLine(seenTag, true);
        return tokens;
    };
    function cleanTripleStache(token) {
        if (token.n.substr(token.n.length - 1) === '}') token.n = token.n.substring(0, token.n.length - 1);
    }
    function trim(s) {
        if (s.trim) return s.trim();
        return s.replace(/^\s*|\s*$/g, '');
    }
    function tagChange(tag, text, index) {
        if (text.charAt(index) != tag.charAt(0)) return false;
        for(var i = 1, l = tag.length; i < l; i++){
            if (text.charAt(index + i) != tag.charAt(i)) return false;
        }
        return true;
    }
    // the tags allowed inside super templates
    var allowedInSuper = {
        '_t': true,
        '\n': true,
        '$': true,
        '/': true
    };
    function buildTree(tokens, kind, stack, customTags) {
        var instructions = [], opener = null, tail = null, token = null;
        tail = stack[stack.length - 1];
        while(tokens.length > 0){
            token = tokens.shift();
            if (tail && tail.tag == '<' && !(token.tag in allowedInSuper)) throw new Error('Illegal content in < super tag.');
            if (Hogan.tags[token.tag] <= Hogan.tags['$'] || isOpener(token, customTags)) {
                stack.push(token);
                token.nodes = buildTree(tokens, token.tag, stack, customTags);
            } else if (token.tag == '/') {
                if (stack.length === 0) throw new Error('Closing tag without opener: /' + token.n);
                opener = stack.pop();
                if (token.n != opener.n && !isCloser(token.n, opener.n, customTags)) throw new Error('Nesting error: ' + opener.n + ' vs. ' + token.n);
                opener.end = token.i;
                return instructions;
            } else if (token.tag == '\n') token.last = tokens.length == 0 || tokens[0].tag == '\n';
            instructions.push(token);
        }
        if (stack.length > 0) throw new Error('missing closing tag: ' + stack.pop().n);
        return instructions;
    }
    function isOpener(token, tags) {
        for(var i = 0, l = tags.length; i < l; i++)if (tags[i].o == token.n) {
            token.tag = '#';
            return true;
        }
    }
    function isCloser(close, open, tags) {
        for(var i = 0, l = tags.length; i < l; i++){
            if (tags[i].c == close && tags[i].o == open) return true;
        }
    }
    function stringifySubstitutions(obj) {
        var items = [];
        for(var key in obj)items.push('"' + esc(key) + '": function(c,p,t,i) {' + obj[key] + '}');
        return "{ " + items.join(",") + " }";
    }
    function stringifyPartials(codeObj) {
        var partials = [];
        for(var key in codeObj.partials)partials.push('"' + esc(key) + '":{name:"' + esc(codeObj.partials[key].name) + '", ' + stringifyPartials(codeObj.partials[key]) + "}");
        return "partials: {" + partials.join(",") + "}, subs: " + stringifySubstitutions(codeObj.subs);
    }
    Hogan.stringify = function(codeObj, text, options) {
        return "{code: function (c,p,i) { " + Hogan.wrapMain(codeObj.code) + " }," + stringifyPartials(codeObj) + "}";
    };
    var serialNo = 0;
    Hogan.generate = function(tree, text, options) {
        serialNo = 0;
        var context = {
            code: '',
            subs: {},
            partials: {}
        };
        Hogan.walk(tree, context);
        if (options.asString) return this.stringify(context, text, options);
        return this.makeTemplate(context, text, options);
    };
    Hogan.wrapMain = function(code) {
        return 'var t=this;t.b(i=i||"");' + code + 'return t.fl();';
    };
    Hogan.template = Hogan.Template;
    Hogan.makeTemplate = function(codeObj, text, options) {
        var template = this.makePartials(codeObj);
        template.code = new Function('c', 'p', 'i', this.wrapMain(codeObj.code));
        return new this.template(template, text, this, options);
    };
    Hogan.makePartials = function(codeObj) {
        var key, template = {
            subs: {},
            partials: codeObj.partials,
            name: codeObj.name
        };
        for(key in template.partials)template.partials[key] = this.makePartials(template.partials[key]);
        for(key in codeObj.subs)template.subs[key] = new Function('c', 'p', 't', 'i', codeObj.subs[key]);
        return template;
    };
    function esc(s) {
        return s.replace(rSlash, '\\\\').replace(rQuot, '\\\"').replace(rNewline, '\\n').replace(rCr, '\\r').replace(rLineSep, '\\u2028').replace(rParagraphSep, '\\u2029');
    }
    function chooseMethod(s) {
        return ~s.indexOf('.') ? 'd' : 'f';
    }
    function createPartial(node, context) {
        var prefix = "<" + (context.prefix || "");
        var sym = prefix + node.n + serialNo++;
        context.partials[sym] = {
            name: node.n,
            partials: {}
        };
        context.code += 't.b(t.rp("' + esc(sym) + '",c,p,"' + (node.indent || '') + '"));';
        return sym;
    }
    Hogan.codegen = {
        '#': function(node, context) {
            context.code += 'if(t.s(t.' + chooseMethod(node.n) + '("' + esc(node.n) + '",c,p,1),' + 'c,p,0,' + node.i + ',' + node.end + ',"' + node.otag + " " + node.ctag + '")){' + 't.rs(c,p,' + 'function(c,p,t){';
            Hogan.walk(node.nodes, context);
            context.code += '});c.pop();}';
        },
        '^': function(node, context) {
            context.code += 'if(!t.s(t.' + chooseMethod(node.n) + '("' + esc(node.n) + '",c,p,1),c,p,1,0,0,"")){';
            Hogan.walk(node.nodes, context);
            context.code += '};';
        },
        '>': createPartial,
        '<': function(node, context) {
            var ctx = {
                partials: {},
                code: '',
                subs: {},
                inPartial: true
            };
            Hogan.walk(node.nodes, ctx);
            var template = context.partials[createPartial(node, context)];
            template.subs = ctx.subs;
            template.partials = ctx.partials;
        },
        '$': function(node, context) {
            var ctx = {
                subs: {},
                code: '',
                partials: context.partials,
                prefix: node.n
            };
            Hogan.walk(node.nodes, ctx);
            context.subs[node.n] = ctx.code;
            if (!context.inPartial) context.code += 't.sub("' + esc(node.n) + '",c,p,i);';
        },
        '\n': function(node, context) {
            context.code += write('"\\n"' + (node.last ? '' : ' + i'));
        },
        '_v': function(node, context) {
            context.code += 't.b(t.v(t.' + chooseMethod(node.n) + '("' + esc(node.n) + '",c,p,0)));';
        },
        '_t': function(node, context) {
            context.code += write('"' + esc(node.text) + '"');
        },
        '{': tripleStache,
        '&': tripleStache
    };
    function tripleStache(node, context) {
        context.code += 't.b(t.t(t.' + chooseMethod(node.n) + '("' + esc(node.n) + '",c,p,0)));';
    }
    function write(s) {
        return 't.b(' + s + ');';
    }
    Hogan.walk = function(nodelist, context) {
        var func;
        for(var i = 0, l = nodelist.length; i < l; i++){
            func = Hogan.codegen[nodelist[i].tag];
            func && func(nodelist[i], context);
        }
        return context;
    };
    Hogan.parse = function(tokens, text, options) {
        options = options || {};
        return buildTree(tokens, '', [], options.sectionTags || []);
    };
    Hogan.cache = {};
    Hogan.cacheKey = function(text, options) {
        return [
            text,
            !!options.asString,
            !!options.disableLambda,
            options.delimiters,
            !!options.modelGet
        ].join('||');
    };
    Hogan.compile = function(text, options) {
        options = options || {};
        var key = Hogan.cacheKey(text, options);
        var template = this.cache[key];
        if (template) {
            var partials = template.partials;
            for(var name in partials)delete partials[name].instance;
            return template;
        }
        template = this.generate(this.parse(this.scan(text, options.delimiters), text, options), text, options);
        return this.cache[key] = template;
    };
})(exports);

},{}],"kCFri":[function(require,module,exports) {
/*
 *  Copyright 2011 Twitter, Inc.
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */ var Hogan = {};
(function(Hogan1) {
    Hogan1.Template = function(codeObj, text, compiler, options) {
        codeObj = codeObj || {};
        this.r = codeObj.code || this.r;
        this.c = compiler;
        this.options = options || {};
        this.text = text || '';
        this.partials = codeObj.partials || {};
        this.subs = codeObj.subs || {};
        this.buf = '';
    };
    Hogan1.Template.prototype = {
        // render: replaced by generated code.
        r: function(context, partials, indent) {
            return '';
        },
        // variable escaping
        v: hoganEscape,
        // triple stache
        t: coerceToString,
        render: function render(context, partials, indent) {
            return this.ri([
                context
            ], partials || {}, indent);
        },
        // render internal -- a hook for overrides that catches partials too
        ri: function(context, partials, indent) {
            return this.r(context, partials, indent);
        },
        // ensurePartial
        ep: function(symbol, partials) {
            var partial = this.partials[symbol];
            // check to see that if we've instantiated this partial before
            var template = partials[partial.name];
            if (partial.instance && partial.base == template) return partial.instance;
            if (typeof template == 'string') {
                if (!this.c) throw new Error("No compiler available.");
                template = this.c.compile(template, this.options);
            }
            if (!template) return null;
            // We use this to check whether the partials dictionary has changed
            this.partials[symbol].base = template;
            if (partial.subs) {
                // Make sure we consider parent template now
                if (!partials.stackText) partials.stackText = {};
                for(key in partial.subs)if (!partials.stackText[key]) partials.stackText[key] = this.activeSub !== undefined && partials.stackText[this.activeSub] ? partials.stackText[this.activeSub] : this.text;
                template = createSpecializedPartial(template, partial.subs, partial.partials, this.stackSubs, this.stackPartials, partials.stackText);
            }
            this.partials[symbol].instance = template;
            return template;
        },
        // tries to find a partial in the current scope and render it
        rp: function(symbol, context, partials, indent) {
            var partial = this.ep(symbol, partials);
            if (!partial) return '';
            return partial.ri(context, partials, indent);
        },
        // render a section
        rs: function(context, partials, section) {
            var tail = context[context.length - 1];
            if (!isArray(tail)) {
                section(context, partials, this);
                return;
            }
            for(var i = 0; i < tail.length; i++){
                context.push(tail[i]);
                section(context, partials, this);
                context.pop();
            }
        },
        // maybe start a section
        s: function(val, ctx, partials, inverted, start, end, tags) {
            var pass;
            if (isArray(val) && val.length === 0) return false;
            if (typeof val == 'function') val = this.ms(val, ctx, partials, inverted, start, end, tags);
            pass = !!val;
            if (!inverted && pass && ctx) ctx.push(typeof val == 'object' ? val : ctx[ctx.length - 1]);
            return pass;
        },
        // find values with dotted names
        d: function(key, ctx, partials, returnFound) {
            var found, names = key.split('.'), val = this.f(names[0], ctx, partials, returnFound), doModelGet = this.options.modelGet, cx = null;
            if (key === '.' && isArray(ctx[ctx.length - 2])) val = ctx[ctx.length - 1];
            else for(var i = 1; i < names.length; i++){
                found = findInScope(names[i], val, doModelGet);
                if (found !== undefined) {
                    cx = val;
                    val = found;
                } else val = '';
            }
            if (returnFound && !val) return false;
            if (!returnFound && typeof val == 'function') {
                ctx.push(cx);
                val = this.mv(val, ctx, partials);
                ctx.pop();
            }
            return val;
        },
        // find values with normal names
        f: function(key, ctx, partials, returnFound) {
            var val = false, v = null, found = false, doModelGet = this.options.modelGet;
            for(var i = ctx.length - 1; i >= 0; i--){
                v = ctx[i];
                val = findInScope(key, v, doModelGet);
                if (val !== undefined) {
                    found = true;
                    break;
                }
            }
            if (!found) return returnFound ? false : "";
            if (!returnFound && typeof val == 'function') val = this.mv(val, ctx, partials);
            return val;
        },
        // higher order templates
        ls: function(func, cx, partials, text, tags) {
            var oldTags = this.options.delimiters;
            this.options.delimiters = tags;
            this.b(this.ct(coerceToString(func.call(cx, text)), cx, partials));
            this.options.delimiters = oldTags;
            return false;
        },
        // compile text
        ct: function(text, cx, partials) {
            if (this.options.disableLambda) throw new Error('Lambda features disabled.');
            return this.c.compile(text, this.options).render(cx, partials);
        },
        // template result buffering
        b: function(s) {
            this.buf += s;
        },
        fl: function() {
            var r = this.buf;
            this.buf = '';
            return r;
        },
        // method replace section
        ms: function(func, ctx, partials, inverted, start, end, tags) {
            var textSource, cx = ctx[ctx.length - 1], result = func.call(cx);
            if (typeof result == 'function') {
                if (inverted) return true;
                else {
                    textSource = this.activeSub && this.subsText && this.subsText[this.activeSub] ? this.subsText[this.activeSub] : this.text;
                    return this.ls(result, cx, partials, textSource.substring(start, end), tags);
                }
            }
            return result;
        },
        // method replace variable
        mv: function(func, ctx, partials) {
            var cx = ctx[ctx.length - 1];
            var result = func.call(cx);
            if (typeof result == 'function') return this.ct(coerceToString(result.call(cx)), cx, partials);
            return result;
        },
        sub: function(name, context, partials, indent) {
            var f = this.subs[name];
            if (f) {
                this.activeSub = name;
                f(context, partials, this, indent);
                this.activeSub = false;
            }
        }
    };
    //Find a key in an object
    function findInScope(key, scope, doModelGet) {
        var val;
        if (scope && typeof scope == 'object') {
            if (scope[key] !== undefined) val = scope[key];
            else if (doModelGet && scope.get && typeof scope.get == 'function') val = scope.get(key);
        }
        return val;
    }
    function createSpecializedPartial(instance, subs, partials, stackSubs, stackPartials, stackText) {
        function PartialTemplate() {}
        PartialTemplate.prototype = instance;
        function Substitutions() {}
        Substitutions.prototype = instance.subs;
        var key;
        var partial = new PartialTemplate();
        partial.subs = new Substitutions();
        partial.subsText = {}; //hehe. substext.
        partial.buf = '';
        stackSubs = stackSubs || {};
        partial.stackSubs = stackSubs;
        partial.subsText = stackText;
        for(key in subs)if (!stackSubs[key]) stackSubs[key] = subs[key];
        for(key in stackSubs)partial.subs[key] = stackSubs[key];
        stackPartials = stackPartials || {};
        partial.stackPartials = stackPartials;
        for(key in partials)if (!stackPartials[key]) stackPartials[key] = partials[key];
        for(key in stackPartials)partial.partials[key] = stackPartials[key];
        return partial;
    }
    var rAmp = /&/g, rLt = /</g, rGt = />/g, rApos = /\'/g, rQuot = /\"/g, hChars = /[&<>\"\']/;
    function coerceToString(val) {
        return String(val === null || val === undefined ? '' : val);
    }
    function hoganEscape(str) {
        str = coerceToString(str);
        return hChars.test(str) ? str.replace(rAmp, '&amp;').replace(rLt, '&lt;').replace(rGt, '&gt;').replace(rApos, '&#39;').replace(rQuot, '&quot;') : str;
    }
    var isArray = Array.isArray || function(a) {
        return Object.prototype.toString.call(a) === '[object Array]';
    };
})(exports);

},{}],"jF2C6":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _getObjectType = require("./getObjectType");
var _getObjectTypeDefault = parcelHelpers.interopDefault(_getObjectType);
function checkRendering(rendering, usage) {
    if (rendering === undefined || typeof rendering !== 'function') throw new Error("The render function is not valid (received type ".concat(_getObjectTypeDefault.default(rendering), ").\n\n").concat(usage));
}
exports.default = checkRendering;

},{"./getObjectType":"3XQ8P","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"3XQ8P":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
function getObjectType(object) {
    return Object.prototype.toString.call(object).slice(8, -1);
}
exports.default = getObjectType;

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"bH0Ll":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "checkIndexUiState", ()=>checkIndexUiState
);
var _capitalize = require("./capitalize");
var _capitalizeDefault = parcelHelpers.interopDefault(_capitalize);
var _logger = require("./logger");
var _typedObject = require("./typedObject"); // Some connectors are responsible for multiple widgets so we need
function _toConsumableArray(arr) {
    return _arrayWithoutHoles(arr) || _iterableToArray(arr) || _unsupportedIterableToArray(arr) || _nonIterableSpread();
}
function _nonIterableSpread() {
    throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
}
function _iterableToArray(iter) {
    if (typeof Symbol !== "undefined" && Symbol.iterator in Object(iter)) return Array.from(iter);
}
function _arrayWithoutHoles(arr) {
    if (Array.isArray(arr)) return _arrayLikeToArray(arr);
}
function _slicedToArray(arr, i) {
    return _arrayWithHoles(arr) || _iterableToArrayLimit(arr, i) || _unsupportedIterableToArray(arr, i) || _nonIterableRest();
}
function _nonIterableRest() {
    throw new TypeError("Invalid attempt to destructure non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
}
function _unsupportedIterableToArray(o, minLen) {
    if (!o) return;
    if (typeof o === "string") return _arrayLikeToArray(o, minLen);
    var n = Object.prototype.toString.call(o).slice(8, -1);
    if (n === "Object" && o.constructor) n = o.constructor.name;
    if (n === "Map" || n === "Set") return Array.from(o);
    if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen);
}
function _arrayLikeToArray(arr, len) {
    if (len == null || len > arr.length) len = arr.length;
    for(var i = 0, arr2 = new Array(len); i < len; i++)arr2[i] = arr[i];
    return arr2;
}
function _iterableToArrayLimit(arr, i) {
    if (typeof Symbol === "undefined" || !(Symbol.iterator in Object(arr))) return;
    var _arr = [];
    var _n = true;
    var _d = false;
    var _e = undefined;
    try {
        for(var _i = arr[Symbol.iterator](), _s; !(_n = (_s = _i.next()).done); _n = true){
            _arr.push(_s.value);
            if (i && _arr.length === i) break;
        }
    } catch (err) {
        _d = true;
        _e = err;
    } finally{
        try {
            if (!_n && _i["return"] != null) _i["return"]();
        } finally{
            if (_d) throw _e;
        }
    }
    return _arr;
}
function _arrayWithHoles(arr) {
    if (Array.isArray(arr)) return arr;
}
// to map them.
function getWidgetNames(connectorName) {
    switch(connectorName){
        case 'range':
            return [];
        case 'menu':
            return [
                'menu',
                'menuSelect'
            ];
        default:
            return [
                connectorName
            ];
    }
}
var stateToWidgetsMap = {
    query: {
        connectors: [
            'connectSearchBox'
        ],
        widgets: [
            'ais.searchBox',
            'ais.autocomplete',
            'ais.voiceSearch'
        ]
    },
    refinementList: {
        connectors: [
            'connectRefinementList'
        ],
        widgets: [
            'ais.refinementList'
        ]
    },
    menu: {
        connectors: [
            'connectMenu'
        ],
        widgets: [
            'ais.menu'
        ]
    },
    hierarchicalMenu: {
        connectors: [
            'connectHierarchicalMenu'
        ],
        widgets: [
            'ais.hierarchicalMenu'
        ]
    },
    numericMenu: {
        connectors: [
            'connectNumericMenu'
        ],
        widgets: [
            'ais.numericMenu'
        ]
    },
    ratingMenu: {
        connectors: [
            'connectRatingMenu'
        ],
        widgets: [
            'ais.ratingMenu'
        ]
    },
    range: {
        connectors: [
            'connectRange'
        ],
        widgets: [
            'ais.rangeInput',
            'ais.rangeSlider',
            'ais.range'
        ]
    },
    toggle: {
        connectors: [
            'connectToggleRefinement'
        ],
        widgets: [
            'ais.toggleRefinement'
        ]
    },
    geoSearch: {
        connectors: [
            'connectGeoSearch'
        ],
        widgets: [
            'ais.geoSearch'
        ]
    },
    sortBy: {
        connectors: [
            'connectSortBy'
        ],
        widgets: [
            'ais.sortBy'
        ]
    },
    page: {
        connectors: [
            'connectPagination'
        ],
        widgets: [
            'ais.pagination',
            'ais.infiniteHits'
        ]
    },
    hitsPerPage: {
        connectors: [
            'connectHitsPerPage'
        ],
        widgets: [
            'ais.hitsPerPage'
        ]
    },
    configure: {
        connectors: [
            'connectConfigure'
        ],
        widgets: [
            'ais.configure'
        ]
    },
    places: {
        connectors: [],
        widgets: [
            'ais.places'
        ]
    }
};
function checkIndexUiState(_ref) {
    var index = _ref.index, indexUiState = _ref.indexUiState;
    var mountedWidgets = index.getWidgets().map(function(widget) {
        return widget.$$type;
    }).filter(Boolean);
    var missingWidgets = _typedObject.keys(indexUiState).reduce(function(acc, parameter) {
        var widgetUiState = stateToWidgetsMap[parameter];
        if (!widgetUiState) return acc;
        var requiredWidgets = widgetUiState.widgets;
        if (requiredWidgets && !requiredWidgets.some(function(requiredWidget) {
            return mountedWidgets.includes(requiredWidget);
        })) acc.push([
            parameter,
            {
                connectors: widgetUiState.connectors,
                widgets: widgetUiState.widgets.map(function(widgetIdentifier) {
                    return widgetIdentifier.split('ais.')[1];
                })
            }
        ]);
        return acc;
    }, []);
    _logger.warning(missingWidgets.length === 0, "The UI state for the index \"".concat(index.getIndexId(), "\" is not consistent with the widgets mounted.\n\nThis can happen when the UI state is specified via `initialUiState`, `routing` or `setUiState` but that the widgets responsible for this state were not added. This results in those query parameters not being sent to the API.\n\nTo fully reflect the state, some widgets need to be added to the index \"").concat(index.getIndexId(), "\":\n\n").concat(missingWidgets.map(function(_ref2) {
        var _ref4;
        var _ref3 = _slicedToArray(_ref2, 2), stateParameter = _ref3[0], widgets = _ref3[1].widgets;
        return "- `".concat(stateParameter, "` needs one of these widgets: ").concat((_ref4 = []).concat.apply(_ref4, _toConsumableArray(widgets.map(function(name) {
            return getWidgetNames(name);
        }))).map(function(name) {
            return "\"".concat(name, "\"");
        }).join(', '));
    }).join('\n'), "\n\nIf you do not wish to display widgets but still want to support their search parameters, you can mount \"virtual widgets\" that don't render anything:\n\n```\n").concat(missingWidgets.filter(function(_ref5) {
        var _ref6 = _slicedToArray(_ref5, 2), _stateParameter = _ref6[0], connectors = _ref6[1].connectors;
        return connectors.length > 0;
    }).map(function(_ref7) {
        var _ref8 = _slicedToArray(_ref7, 2), _stateParameter = _ref8[0], _ref8$ = _ref8[1], connectors = _ref8$.connectors, widgets = _ref8$.widgets;
        var capitalizedWidget = _capitalizeDefault.default(widgets[0]);
        var connectorName = connectors[0];
        return "const virtual".concat(capitalizedWidget, " = ").concat(connectorName, "(() => null);");
    }).join('\n'), "\n\nsearch.addWidgets([\n  ").concat(missingWidgets.filter(function(_ref9) {
        var _ref10 = _slicedToArray(_ref9, 2), _stateParameter = _ref10[0], connectors = _ref10[1].connectors;
        return connectors.length > 0;
    }).map(function(_ref11) {
        var _ref12 = _slicedToArray(_ref11, 2), _stateParameter = _ref12[0], widgets = _ref12[1].widgets;
        var capitalizedWidget = _capitalizeDefault.default(widgets[0]);
        return "virtual".concat(capitalizedWidget, "({ /* ... */ })");
    }).join(',\n  '), "\n]);\n```\n\nIf you're using custom widgets that do set these query parameters, we recommend using connectors instead.\n\nSee https://www.algolia.com/doc/guides/building-search-ui/widgets/customize-an-existing-widget/js/#customize-the-complete-ui-of-the-widgets"));
}

},{"./capitalize":"1J2wi","./logger":"glTTt","./typedObject":"5SpMp","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"glTTt":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "warn", ()=>warn
);
parcelHelpers.export(exports, "deprecate", ()=>deprecate
);
parcelHelpers.export(exports, "warning", ()=>_warning
);
var _noop = require("./noop");
var _noopDefault = parcelHelpers.interopDefault(_noop);
/**
 * Logs a warning when this function is called, in development environment only.
 */ var deprecate = function deprecate(fn) {
    return fn;
};
/**
 * Logs a warning
 * This is used to log issues in development environment only.
 */ var warn = _noopDefault.default;
/**
 * Logs a warning if the condition is not met.
 * This is used to log issues in development environment only.
 */ var _warning = _noopDefault.default;
warn = function warn(message) {
    // eslint-disable-next-line no-console
    console.warn("[InstantSearch.js]: ".concat(message.trim()));
};
deprecate = function deprecate(fn, message) {
    var hasAlreadyPrinted = false;
    return function() {
        if (!hasAlreadyPrinted) {
            hasAlreadyPrinted = true;
            warn(message);
        }
        return fn.apply(void 0, arguments);
    };
};
_warning = function warning(condition, message) {
    if (condition) return;
    var hasAlreadyPrinted = _warning.cache[message];
    if (!hasAlreadyPrinted) {
        _warning.cache[message] = true;
        warn(message);
    }
};
_warning.cache = {};

},{"./noop":"6iazv","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"6iazv":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
function noop() {}
exports.default = noop;

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"5SpMp":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "keys", ()=>keys
);
var keys = Object.keys;

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"2Q0lT":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
function getPropertyByPath(object, path) {
    var parts = Array.isArray(path) ? path : path.split('.');
    return parts.reduce(function(current, key) {
        return current && current[key];
    }, object);
}
exports.default = getPropertyByPath;

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"gQhvL":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
// This is the `Number.isFinite()` polyfill recommended by MDN.
// We do not provide any tests for this function.
// See: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Number/isFinite#Polyfill
function isFiniteNumber(value) {
    return typeof value === 'number' && isFinite(value);
}
exports.default = isFiniteNumber;

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"cIivc":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
function _typeof(obj1) {
    if (typeof Symbol === "function" && typeof Symbol.iterator === "symbol") _typeof = function _typeof(obj) {
        return typeof obj;
    };
    else _typeof = function _typeof(obj) {
        return obj && typeof Symbol === "function" && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj;
    };
    return _typeof(obj1);
}
/**
 * This implementation is taken from Lodash implementation.
 * See: https://github.com/lodash/lodash/blob/master/isPlainObject.js
 */ function getTag(value) {
    if (value === null) return value === undefined ? '[object Undefined]' : '[object Null]';
    return Object.prototype.toString.call(value);
}
function isObjectLike(value) {
    return _typeof(value) === 'object' && value !== null;
}
/**
 * Checks if `value` is a plain object.
 *
 * A plain object is an object created by the `Object`
 * constructor or with a `[[Prototype]]` of `null`.
 */ function isPlainObject(value) {
    if (!isObjectLike(value) || getTag(value) !== '[object Object]') return false;
    if (Object.getPrototypeOf(value) === null) return true;
    var proto = value;
    while(Object.getPrototypeOf(proto) !== null)proto = Object.getPrototypeOf(proto);
    return Object.getPrototypeOf(value) === proto;
}
exports.default = isPlainObject;

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"1dHGc":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
function _toConsumableArray(arr) {
    return _arrayWithoutHoles(arr) || _iterableToArray(arr) || _unsupportedIterableToArray(arr) || _nonIterableSpread();
}
function _nonIterableSpread() {
    throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
}
function _unsupportedIterableToArray(o, minLen) {
    if (!o) return;
    if (typeof o === "string") return _arrayLikeToArray(o, minLen);
    var n = Object.prototype.toString.call(o).slice(8, -1);
    if (n === "Object" && o.constructor) n = o.constructor.name;
    if (n === "Map" || n === "Set") return Array.from(o);
    if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen);
}
function _iterableToArray(iter) {
    if (typeof Symbol !== "undefined" && Symbol.iterator in Object(iter)) return Array.from(iter);
}
function _arrayWithoutHoles(arr) {
    if (Array.isArray(arr)) return _arrayLikeToArray(arr);
}
function _arrayLikeToArray(arr, len) {
    if (len == null || len > arr.length) len = arr.length;
    for(var i = 0, arr2 = new Array(len); i < len; i++)arr2[i] = arr[i];
    return arr2;
}
function range(_ref) {
    var _ref$start = _ref.start, start = _ref$start === void 0 ? 0 : _ref$start, end = _ref.end, _ref$step = _ref.step, step = _ref$step === void 0 ? 1 : _ref$step;
    // We can't divide by 0 so we re-assign the step to 1 if it happens.
    var limitStep = step === 0 ? 1 : step; // In some cases the array to create has a decimal length.
    // We therefore need to round the value.
    // Example:
    //   { start: 1, end: 5000, step: 500 }
    //   => Array length = (5000 - 1) / 500 = 9.998
    var arrayLength = Math.round((end - start) / limitStep);
    return _toConsumableArray(Array(arrayLength)).map(function(_, current) {
        return start + current * limitStep;
    });
}
exports.default = range;

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"14V8N":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
function isPrimitive(obj) {
    return obj !== Object(obj);
}
function isEqual(first, second) {
    if (first === second) return true;
    if (isPrimitive(first) || isPrimitive(second) || typeof first === 'function' || typeof second === 'function') return first === second;
    if (Object.keys(first).length !== Object.keys(second).length) return false;
    for(var _i = 0, _Object$keys = Object.keys(first); _i < _Object$keys.length; _i++){
        var key = _Object$keys[_i];
        if (!(key in second)) return false;
        if (!isEqual(first[key], second[key])) return false;
    }
    return true;
}
exports.default = isEqual;

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"eLn1u":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/**
 * This implementation is taken from Lodash implementation.
 * See: https://github.com/lodash/lodash/blob/4.17.11-npm/escape.js
 */ // Used to map characters to HTML entities.
var htmlEscapes = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
}; // Used to match HTML entities and HTML characters.
var regexUnescapedHtml = /[&<>"']/g;
var regexHasUnescapedHtml = RegExp(regexUnescapedHtml.source);
/**
 * Converts the characters "&", "<", ">", '"', and "'" in `string` to their
 * corresponding HTML entities.
 */ function escape(value) {
    return value && regexHasUnescapedHtml.test(value) ? value.replace(regexUnescapedHtml, function(character) {
        return htmlEscapes[character];
    }) : value;
}
exports.default = escape;

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"cWsJ0":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/**
 * This implementation is taken from Lodash implementation.
 * See: https://github.com/lodash/lodash/blob/4.17.11-npm/unescape.js
 */ // Used to map HTML entities to characters.
var htmlEscapes = {
    '&amp;': '&',
    '&lt;': '<',
    '&gt;': '>',
    '&quot;': '"',
    '&#39;': "'"
}; // Used to match HTML entities and HTML characters.
var regexEscapedHtml = /&(amp|quot|lt|gt|#39);/g;
var regexHasEscapedHtml = RegExp(regexEscapedHtml.source);
function unescape(value) {
    return value && regexHasEscapedHtml.test(value) ? value.replace(regexEscapedHtml, function(character) {
        return htmlEscapes[character];
    }) : value;
}
exports.default = unescape;

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"12qh7":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _escapeHighlight = require("./escape-highlight");
function concatHighlightedParts(parts) {
    var highlightPreTag = _escapeHighlight.TAG_REPLACEMENT.highlightPreTag, highlightPostTag = _escapeHighlight.TAG_REPLACEMENT.highlightPostTag;
    return parts.map(function(part) {
        return part.isHighlighted ? highlightPreTag + part.value + highlightPostTag : part.value;
    }).join('');
}
exports.default = concatHighlightedParts;

},{"./escape-highlight":"59mW0","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"59mW0":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "TAG_PLACEHOLDER", ()=>TAG_PLACEHOLDER
);
parcelHelpers.export(exports, "TAG_REPLACEMENT", ()=>TAG_REPLACEMENT
);
parcelHelpers.export(exports, "escapeHits", ()=>escapeHits
);
parcelHelpers.export(exports, "escapeFacets", ()=>escapeFacets
);
var _escape = require("./escape");
var _escapeDefault = parcelHelpers.interopDefault(_escape);
var _isPlainObject = require("./isPlainObject");
var _isPlainObjectDefault = parcelHelpers.interopDefault(_isPlainObject);
function _extends() {
    _extends = Object.assign || function(target) {
        for(var i = 1; i < arguments.length; i++){
            var source = arguments[i];
            for(var key in source)if (Object.prototype.hasOwnProperty.call(source, key)) target[key] = source[key];
        }
        return target;
    };
    return _extends.apply(this, arguments);
}
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        if (enumerableOnly) symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        });
        keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = arguments[i] != null ? arguments[i] : {};
        if (i % 2) ownKeys(Object(source), true).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        });
        else if (Object.getOwnPropertyDescriptors) Object.defineProperties(target, Object.getOwnPropertyDescriptors(source));
        else ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
var TAG_PLACEHOLDER = {
    highlightPreTag: '__ais-highlight__',
    highlightPostTag: '__/ais-highlight__'
};
var TAG_REPLACEMENT = {
    highlightPreTag: '<mark>',
    highlightPostTag: '</mark>'
};
function replaceTagsAndEscape(value) {
    return _escapeDefault.default(value).replace(new RegExp(TAG_PLACEHOLDER.highlightPreTag, 'g'), TAG_REPLACEMENT.highlightPreTag).replace(new RegExp(TAG_PLACEHOLDER.highlightPostTag, 'g'), TAG_REPLACEMENT.highlightPostTag);
}
function recursiveEscape(input) {
    if (_isPlainObjectDefault.default(input) && typeof input.value !== 'string') return Object.keys(input).reduce(function(acc, key) {
        return _objectSpread(_objectSpread({}, acc), {}, _defineProperty({}, key, recursiveEscape(input[key])));
    }, {});
    if (Array.isArray(input)) return input.map(recursiveEscape);
    return _objectSpread(_objectSpread({}, input), {}, {
        value: replaceTagsAndEscape(input.value)
    });
}
function escapeHits(hits) {
    if (hits.__escaped === undefined) {
        // We don't override the value on hit because it will mutate the raw results
        // instead we make a shallow copy and we assign the escaped values on it.
        hits = hits.map(function(_ref) {
            var hit = _extends({}, _ref);
            if (hit._highlightResult) hit._highlightResult = recursiveEscape(hit._highlightResult);
            if (hit._snippetResult) hit._snippetResult = recursiveEscape(hit._snippetResult);
            return hit;
        });
        hits.__escaped = true;
    }
    return hits;
}
function escapeFacets(facetHits) {
    return facetHits.map(function(h) {
        return _objectSpread(_objectSpread({}, h), {}, {
            highlighted: replaceTagsAndEscape(h.highlighted)
        });
    });
}

},{"./escape":"eLn1u","./isPlainObject":"cIivc","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"7jCC9":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _escapeHighlight = require("./escape-highlight");
function getHighlightedParts(highlightedValue) {
    var highlightPostTag = _escapeHighlight.TAG_REPLACEMENT.highlightPostTag, highlightPreTag = _escapeHighlight.TAG_REPLACEMENT.highlightPreTag;
    var splitByPreTag = highlightedValue.split(highlightPreTag);
    var firstValue = splitByPreTag.shift();
    var elements = !firstValue ? [] : [
        {
            value: firstValue,
            isHighlighted: false
        }
    ];
    splitByPreTag.forEach(function(split) {
        var splitByPostTag = split.split(highlightPostTag);
        elements.push({
            value: splitByPostTag[0],
            isHighlighted: true
        });
        if (splitByPostTag[1] !== '') elements.push({
            value: splitByPostTag[1],
            isHighlighted: false
        });
    });
    return elements;
}
exports.default = getHighlightedParts;

},{"./escape-highlight":"59mW0","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"fAPc5":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _unescape = require("./unescape");
var _unescapeDefault = parcelHelpers.interopDefault(_unescape);
var hasAlphanumeric = new RegExp(/\w/i);
function getHighlightFromSiblings(parts, i) {
    var _parts, _parts2;
    var current = parts[i];
    var isNextHighlighted = ((_parts = parts[i + 1]) === null || _parts === void 0 ? void 0 : _parts.isHighlighted) || true;
    var isPreviousHighlighted = ((_parts2 = parts[i - 1]) === null || _parts2 === void 0 ? void 0 : _parts2.isHighlighted) || true;
    if (!hasAlphanumeric.test(_unescapeDefault.default(current.value)) && isPreviousHighlighted === isNextHighlighted) return isPreviousHighlighted;
    return current.isHighlighted;
}
exports.default = getHighlightFromSiblings;

},{"./unescape":"cWsJ0","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"kBcoN":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _getHighlightFromSiblings = require("./getHighlightFromSiblings");
var _getHighlightFromSiblingsDefault = parcelHelpers.interopDefault(_getHighlightFromSiblings);
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        if (enumerableOnly) symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        });
        keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = arguments[i] != null ? arguments[i] : {};
        if (i % 2) ownKeys(Object(source), true).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        });
        else if (Object.getOwnPropertyDescriptors) Object.defineProperties(target, Object.getOwnPropertyDescriptors(source));
        else ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
function reverseHighlightedParts(parts) {
    if (!parts.some(function(part) {
        return part.isHighlighted;
    })) return parts.map(function(part) {
        return _objectSpread(_objectSpread({}, part), {}, {
            isHighlighted: false
        });
    });
    return parts.map(function(part, i) {
        return _objectSpread(_objectSpread({}, part), {}, {
            isHighlighted: !_getHighlightFromSiblingsDefault.default(parts, i)
        });
    });
}
exports.default = reverseHighlightedParts;

},{"./getHighlightFromSiblings":"fAPc5","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"6Dhef":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
// We aren't using the native `Array.prototype.find` because the refactor away from Lodash is not
// published as a major version.
// Relying on the `find` polyfill on user-land, which before was only required for niche use-cases,
// was decided as too risky.
// @MAJOR Replace with the native `Array.prototype.find` method
// https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/find
function find(items, predicate) {
    var value;
    for(var i = 0; i < items.length; i++){
        value = items[i]; // inlined for performance: if (Call(predicate, thisArg, [value, i, list])) {
        if (predicate(value, i, items)) return value;
    }
    return undefined;
}
exports.default = find;

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"8tlAy":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
// We aren't using the native `Array.prototype.findIndex` because the refactor away from Lodash is not
// published as a major version.
// Relying on the `findIndex` polyfill on user-land, which before was only required for niche use-cases,
// was decided as too risky.
// @MAJOR Replace with the native `Array.prototype.findIndex` method
// https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/findIndex
function findIndex(array, comparator) {
    if (!Array.isArray(array)) return -1;
    for(var i = 0; i < array.length; i++){
        if (comparator(array[i])) return i;
    }
    return -1;
}
exports.default = findIndex;

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"9Li6L":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _findIndex = require("./findIndex");
var _findIndexDefault = parcelHelpers.interopDefault(_findIndex);
var _uniq = require("./uniq");
var _uniqDefault = parcelHelpers.interopDefault(_uniq);
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        if (enumerableOnly) symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        });
        keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = arguments[i] != null ? arguments[i] : {};
        if (i % 2) ownKeys(Object(source), true).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        });
        else if (Object.getOwnPropertyDescriptors) Object.defineProperties(target, Object.getOwnPropertyDescriptors(source));
        else ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
function _objectWithoutProperties(source, excluded) {
    if (source == null) return {};
    var target = _objectWithoutPropertiesLoose(source, excluded);
    var key, i;
    if (Object.getOwnPropertySymbols) {
        var sourceSymbolKeys = Object.getOwnPropertySymbols(source);
        for(i = 0; i < sourceSymbolKeys.length; i++){
            key = sourceSymbolKeys[i];
            if (excluded.indexOf(key) >= 0) continue;
            if (!Object.prototype.propertyIsEnumerable.call(source, key)) continue;
            target[key] = source[key];
        }
    }
    return target;
}
function _objectWithoutPropertiesLoose(source, excluded) {
    if (source == null) return {};
    var target = {};
    var sourceKeys = Object.keys(source);
    var key, i;
    for(i = 0; i < sourceKeys.length; i++){
        key = sourceKeys[i];
        if (excluded.indexOf(key) >= 0) continue;
        target[key] = source[key];
    }
    return target;
}
var mergeWithRest = function mergeWithRest(left, right) {
    var facets = right.facets, disjunctiveFacets = right.disjunctiveFacets, facetsRefinements = right.facetsRefinements, facetsExcludes = right.facetsExcludes, disjunctiveFacetsRefinements = right.disjunctiveFacetsRefinements, numericRefinements = right.numericRefinements, tagRefinements = right.tagRefinements, hierarchicalFacets = right.hierarchicalFacets, hierarchicalFacetsRefinements = right.hierarchicalFacetsRefinements, ruleContexts = right.ruleContexts, rest = _objectWithoutProperties(right, [
        "facets",
        "disjunctiveFacets",
        "facetsRefinements",
        "facetsExcludes",
        "disjunctiveFacetsRefinements",
        "numericRefinements",
        "tagRefinements",
        "hierarchicalFacets",
        "hierarchicalFacetsRefinements",
        "ruleContexts"
    ]);
    return left.setQueryParameters(rest);
}; // Merge facets
var mergeFacets = function mergeFacets(left, right) {
    return right.facets.reduce(function(_, name) {
        return _.addFacet(name);
    }, left);
};
var mergeDisjunctiveFacets = function mergeDisjunctiveFacets(left, right) {
    return right.disjunctiveFacets.reduce(function(_, name) {
        return _.addDisjunctiveFacet(name);
    }, left);
};
var mergeHierarchicalFacets = function mergeHierarchicalFacets(left, right) {
    return left.setQueryParameters({
        hierarchicalFacets: right.hierarchicalFacets.reduce(function(facets, facet) {
            var index = _findIndexDefault.default(facets, function(_) {
                return _.name === facet.name;
            });
            if (index === -1) return facets.concat(facet);
            var nextFacets = facets.slice();
            nextFacets.splice(index, 1, facet);
            return nextFacets;
        }, left.hierarchicalFacets)
    });
}; // Merge facet refinements
var mergeTagRefinements = function mergeTagRefinements(left, right) {
    return right.tagRefinements.reduce(function(_, value) {
        return _.addTagRefinement(value);
    }, left);
};
var mergeFacetRefinements = function mergeFacetRefinements(left, right) {
    return left.setQueryParameters({
        facetsRefinements: _objectSpread(_objectSpread({}, left.facetsRefinements), right.facetsRefinements)
    });
};
var mergeFacetsExcludes = function mergeFacetsExcludes(left, right) {
    return left.setQueryParameters({
        facetsExcludes: _objectSpread(_objectSpread({}, left.facetsExcludes), right.facetsExcludes)
    });
};
var mergeDisjunctiveFacetsRefinements = function mergeDisjunctiveFacetsRefinements(left, right) {
    return left.setQueryParameters({
        disjunctiveFacetsRefinements: _objectSpread(_objectSpread({}, left.disjunctiveFacetsRefinements), right.disjunctiveFacetsRefinements)
    });
};
var mergeNumericRefinements = function mergeNumericRefinements(left, right) {
    return left.setQueryParameters({
        numericRefinements: _objectSpread(_objectSpread({}, left.numericRefinements), right.numericRefinements)
    });
};
var mergeHierarchicalFacetsRefinements = function mergeHierarchicalFacetsRefinements(left, right) {
    return left.setQueryParameters({
        hierarchicalFacetsRefinements: _objectSpread(_objectSpread({}, left.hierarchicalFacetsRefinements), right.hierarchicalFacetsRefinements)
    });
};
var mergeRuleContexts = function mergeRuleContexts(left, right) {
    var ruleContexts = _uniqDefault.default([].concat(left.ruleContexts).concat(right.ruleContexts).filter(Boolean));
    if (ruleContexts.length > 0) return left.setQueryParameters({
        ruleContexts: ruleContexts
    });
    return left;
};
var merge = function merge() {
    for(var _len = arguments.length, parameters = new Array(_len), _key = 0; _key < _len; _key++)parameters[_key] = arguments[_key];
    return parameters.reduce(function(left, right) {
        var hierarchicalFacetsRefinementsMerged = mergeHierarchicalFacetsRefinements(left, right);
        var hierarchicalFacetsMerged = mergeHierarchicalFacets(hierarchicalFacetsRefinementsMerged, right);
        var tagRefinementsMerged = mergeTagRefinements(hierarchicalFacetsMerged, right);
        var numericRefinementsMerged = mergeNumericRefinements(tagRefinementsMerged, right);
        var disjunctiveFacetsRefinementsMerged = mergeDisjunctiveFacetsRefinements(numericRefinementsMerged, right);
        var facetsExcludesMerged = mergeFacetsExcludes(disjunctiveFacetsRefinementsMerged, right);
        var facetRefinementsMerged = mergeFacetRefinements(facetsExcludesMerged, right);
        var disjunctiveFacetsMerged = mergeDisjunctiveFacets(facetRefinementsMerged, right);
        var ruleContextsMerged = mergeRuleContexts(disjunctiveFacetsMerged, right);
        var facetsMerged = mergeFacets(ruleContextsMerged, right);
        return mergeWithRest(facetsMerged, right);
    });
};
exports.default = merge;

},{"./findIndex":"8tlAy","./uniq":"2Q0ce","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"a7lVI":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var resolveSearchParameters = function resolveSearchParameters(current) {
    var parent = current.getParent();
    var states = [
        current.getHelper().state
    ];
    while(parent !== null){
        states = [
            parent.getHelper().state
        ].concat(states);
        parent = parent.getParent();
    }
    return states;
};
exports.default = resolveSearchParameters;

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"gLqHy":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "createDocumentationLink", ()=>createDocumentationLink
);
parcelHelpers.export(exports, "createDocumentationMessageGenerator", ()=>createDocumentationMessageGenerator
);
var createDocumentationLink = function createDocumentationLink(_ref) {
    var name = _ref.name, _ref$connector = _ref.connector, connector = _ref$connector === void 0 ? false : _ref$connector;
    return [
        'https://www.algolia.com/doc/api-reference/widgets/',
        name,
        '/js/',
        connector ? '#connector' : ''
    ].join('');
};
var createDocumentationMessageGenerator = function createDocumentationMessageGenerator() {
    for(var _len = arguments.length, widgets = new Array(_len), _key = 0; _key < _len; _key++)widgets[_key] = arguments[_key];
    var links = widgets.map(function(widget) {
        return createDocumentationLink(widget);
    }).join(', ');
    return function(message) {
        return [
            message,
            "See documentation: ".concat(links)
        ].filter(Boolean).join('\n\n');
    };
};

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"dMQpP":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "addAbsolutePosition", ()=>addAbsolutePosition
);
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        if (enumerableOnly) symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        });
        keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = arguments[i] != null ? arguments[i] : {};
        if (i % 2) ownKeys(Object(source), true).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        });
        else if (Object.getOwnPropertyDescriptors) Object.defineProperties(target, Object.getOwnPropertyDescriptors(source));
        else ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
var addAbsolutePosition = function addAbsolutePosition(hits, page, hitsPerPage) {
    return hits.map(function(hit, idx) {
        return _objectSpread(_objectSpread({}, hit), {}, {
            __position: hitsPerPage * page + idx + 1
        });
    });
};

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"iBpEo":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "addQueryID", ()=>addQueryID
);
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        if (enumerableOnly) symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        });
        keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = arguments[i] != null ? arguments[i] : {};
        if (i % 2) ownKeys(Object(source), true).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        });
        else if (Object.getOwnPropertyDescriptors) Object.defineProperties(target, Object.getOwnPropertyDescriptors(source));
        else ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
function addQueryID(hits, queryID) {
    if (!queryID) return hits;
    return hits.map(function(hit) {
        return _objectSpread(_objectSpread({}, hit), {}, {
            __queryID: queryID
        });
    });
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"b5SV4":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
function isFacetRefined(helper, facet, value) {
    if (helper.state.isHierarchicalFacet(facet)) return helper.state.isHierarchicalFacetRefined(facet, value);
    else if (helper.state.isConjunctiveFacet(facet)) return helper.state.isFacetRefined(facet, value);
    else return helper.state.isDisjunctiveFacetRefined(facet, value);
}
exports.default = isFacetRefined;

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"05go2":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "createSendEventForFacet", ()=>createSendEventForFacet
);
var _isFacetRefined = require("./isFacetRefined");
var _isFacetRefinedDefault = parcelHelpers.interopDefault(_isFacetRefined);
function _typeof(obj1) {
    if (typeof Symbol === "function" && typeof Symbol.iterator === "symbol") _typeof = function _typeof(obj) {
        return typeof obj;
    };
    else _typeof = function _typeof(obj) {
        return obj && typeof Symbol === "function" && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj;
    };
    return _typeof(obj1);
}
function createSendEventForFacet(_ref) {
    var instantSearchInstance = _ref.instantSearchInstance, helper = _ref.helper, attribute = _ref.attribute, widgetType = _ref.widgetType;
    var sendEventForFacet = function sendEventForFacet() {
        for(var _len = arguments.length, args = new Array(_len), _key = 0; _key < _len; _key++)args[_key] = arguments[_key];
        var eventType = args[0], facetValue = args[1], _args$ = args[2], eventName = _args$ === void 0 ? 'Filter Applied' : _args$;
        if (args.length === 1 && _typeof(args[0]) === 'object') instantSearchInstance.sendEventToInsights(args[0]);
        else if (eventType === 'click' && (args.length === 2 || args.length === 3)) {
            if (!_isFacetRefinedDefault.default(helper, attribute, facetValue)) // send event only when the facet is being checked "ON"
            instantSearchInstance.sendEventToInsights({
                insightsMethod: 'clickedFilters',
                widgetType: widgetType,
                eventType: eventType,
                payload: {
                    eventName: eventName,
                    index: helper.getIndex(),
                    filters: [
                        "".concat(attribute, ":").concat(facetValue)
                    ]
                },
                attribute: attribute
            });
        } else throw new Error("You need to pass two arguments like:\n  sendEvent('click', facetValue);\n\nIf you want to send a custom payload, you can pass one object: sendEvent(customPayload);\n");
    };
    return sendEventForFacet;
}

},{"./isFacetRefined":"b5SV4","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"24sIF":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "createSendEventForHits", ()=>createSendEventForHits
);
parcelHelpers.export(exports, "createBindEventForHits", ()=>createBindEventForHits
);
var _serializer = require("../../lib/utils/serializer");
function _typeof(obj1) {
    if (typeof Symbol === "function" && typeof Symbol.iterator === "symbol") _typeof = function _typeof(obj) {
        return typeof obj;
    };
    else _typeof = function _typeof(obj) {
        return obj && typeof Symbol === "function" && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj;
    };
    return _typeof(obj1);
}
var buildPayload = function buildPayload(_ref) {
    var index = _ref.index, widgetType = _ref.widgetType, methodName = _ref.methodName, args = _ref.args;
    if (args.length === 1 && _typeof(args[0]) === 'object') return args[0];
    var eventType = args[0];
    var hits = args[1];
    var eventName = args[2];
    if (!hits) throw new Error("You need to pass hit or hits as the second argument like:\n  ".concat(methodName, "(eventType, hit);\n  "));
    if ((eventType === 'click' || eventType === 'conversion') && !eventName) throw new Error("You need to pass eventName as the third argument for 'click' or 'conversion' events like:\n  ".concat(methodName, "('click', hit, 'Product Purchased');\n\n  To learn more about event naming: https://www.algolia.com/doc/guides/getting-insights-and-analytics/search-analytics/click-through-and-conversions/in-depth/clicks-conversions-best-practices/\n  "));
    var hitsArray = Array.isArray(hits) ? removeEscapedFromHits(hits) : [
        hits
    ];
    if (hitsArray.length === 0) return null;
    var queryID = hitsArray[0].__queryID;
    var objectIDs = hitsArray.map(function(hit) {
        return hit.objectID;
    });
    var positions = hitsArray.map(function(hit) {
        return hit.__position;
    });
    if (eventType === 'view') return {
        insightsMethod: 'viewedObjectIDs',
        widgetType: widgetType,
        eventType: eventType,
        payload: {
            eventName: eventName || 'Hits Viewed',
            index: index,
            objectIDs: objectIDs
        },
        hits: hitsArray
    };
    else if (eventType === 'click') return {
        insightsMethod: 'clickedObjectIDsAfterSearch',
        widgetType: widgetType,
        eventType: eventType,
        payload: {
            eventName: eventName,
            index: index,
            queryID: queryID,
            objectIDs: objectIDs,
            positions: positions
        },
        hits: hitsArray
    };
    else if (eventType === 'conversion') return {
        insightsMethod: 'convertedObjectIDsAfterSearch',
        widgetType: widgetType,
        eventType: eventType,
        payload: {
            eventName: eventName,
            index: index,
            queryID: queryID,
            objectIDs: objectIDs
        },
        hits: hitsArray
    };
    else throw new Error("eventType(\"".concat(eventType, "\") is not supported.\n    If you want to send a custom payload, you can pass one object: ").concat(methodName, "(customPayload);\n    "));
};
function removeEscapedFromHits(hits) {
    // this returns without `hits.__escaped`
    // and this way it doesn't mutate the original `hits`
    return hits.map(function(hit) {
        return hit;
    });
}
function createSendEventForHits(_ref2) {
    var instantSearchInstance = _ref2.instantSearchInstance, index = _ref2.index, widgetType = _ref2.widgetType;
    var sendEventForHits = function sendEventForHits() {
        for(var _len = arguments.length, args = new Array(_len), _key = 0; _key < _len; _key++)args[_key] = arguments[_key];
        var payload = buildPayload({
            widgetType: widgetType,
            index: index,
            methodName: 'sendEvent',
            args: args
        });
        if (payload) instantSearchInstance.sendEventToInsights(payload);
    };
    return sendEventForHits;
}
function createBindEventForHits(_ref3) {
    var index = _ref3.index, widgetType = _ref3.widgetType;
    var bindEventForHits = function bindEventForHits() {
        for(var _len2 = arguments.length, args = new Array(_len2), _key2 = 0; _key2 < _len2; _key2++)args[_key2] = arguments[_key2];
        var payload = buildPayload({
            widgetType: widgetType,
            index: index,
            methodName: 'bindEvent',
            args: args
        });
        return payload ? "data-insights-event=".concat(_serializer.serializePayload(payload)) : '';
    };
    return bindEventForHits;
}

},{"../../lib/utils/serializer":"jg61H","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"jg61H":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "serializePayload", ()=>serializePayload
);
parcelHelpers.export(exports, "deserializePayload", ()=>deserializePayload
);
function serializePayload(payload) {
    return btoa(encodeURIComponent(JSON.stringify(payload)));
}
function deserializePayload(payload) {
    return JSON.parse(decodeURIComponent(atob(payload)));
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"ekxD4":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "convertNumericRefinementsToFilters", ()=>convertNumericRefinementsToFilters
);
function convertNumericRefinementsToFilters(state, attribute) {
    if (!state) return null;
    var filtersObj = state.numericRefinements[attribute];
    /*
    filtersObj === {
      "<=": [10],
      "=": [],
      ">=": [5]
    }
  */ var filters = [];
    Object.keys(filtersObj).filter(function(operator) {
        return Array.isArray(filtersObj[operator]) && filtersObj[operator].length > 0;
    }).forEach(function(operator) {
        filtersObj[operator].forEach(function(value) {
            filters.push("".concat(attribute).concat(operator).concat(value));
        });
    });
    return filters;
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"hkkLK":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
exports.default = '4.22.0';

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"8IHo3":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _helpers = require("../helpers");
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        if (enumerableOnly) symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        });
        keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = arguments[i] != null ? arguments[i] : {};
        if (i % 2) ownKeys(Object(source), true).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        });
        else if (Object.getOwnPropertyDescriptors) Object.defineProperties(target, Object.getOwnPropertyDescriptors(source));
        else ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
function hoganHelpers(_ref) {
    var numberLocale = _ref.numberLocale;
    return {
        formatNumber: function formatNumber(value, render) {
            return Number(render(value)).toLocaleString(numberLocale);
        },
        highlight: function highlight(options, render) {
            try {
                var highlightOptions = JSON.parse(options);
                return render(_helpers.highlight(_objectSpread(_objectSpread({}, highlightOptions), {}, {
                    hit: this
                })));
            } catch (error) {
                throw new Error("\nThe highlight helper expects a JSON object of the format:\n{ \"attribute\": \"name\", \"highlightedTagName\": \"mark\" }");
            }
        },
        reverseHighlight: function reverseHighlight(options, render) {
            try {
                var reverseHighlightOptions = JSON.parse(options);
                return render(_helpers.reverseHighlight(_objectSpread(_objectSpread({}, reverseHighlightOptions), {}, {
                    hit: this
                })));
            } catch (error) {
                throw new Error("\n  The reverseHighlight helper expects a JSON object of the format:\n  { \"attribute\": \"name\", \"highlightedTagName\": \"mark\" }");
            }
        },
        snippet: function snippet(options, render) {
            try {
                var snippetOptions = JSON.parse(options);
                return render(_helpers.snippet(_objectSpread(_objectSpread({}, snippetOptions), {}, {
                    hit: this
                })));
            } catch (error) {
                throw new Error("\nThe snippet helper expects a JSON object of the format:\n{ \"attribute\": \"name\", \"highlightedTagName\": \"mark\" }");
            }
        },
        reverseSnippet: function reverseSnippet(options, render) {
            try {
                var reverseSnippetOptions = JSON.parse(options);
                return render(_helpers.reverseSnippet(_objectSpread(_objectSpread({}, reverseSnippetOptions), {}, {
                    hit: this
                })));
            } catch (error) {
                throw new Error("\n  The reverseSnippet helper expects a JSON object of the format:\n  { \"attribute\": \"name\", \"highlightedTagName\": \"mark\" }");
            }
        },
        insights: function insights(options, render) {
            try {
                var _JSON$parse = JSON.parse(options), method = _JSON$parse.method, payload = _JSON$parse.payload;
                return render(_helpers.insights(method, _objectSpread({
                    objectIDs: [
                        this.objectID
                    ]
                }, payload)));
            } catch (error) {
                throw new Error("\nThe insights helper expects a JSON object of the format:\n{ \"method\": \"method-name\", \"payload\": { \"eventName\": \"name of the event\" } }");
            }
        }
    };
}
exports.default = hoganHelpers;

},{"../helpers":"8kgzi","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"8kgzi":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "reverseHighlight", ()=>_reverseHighlightDefault.default
);
parcelHelpers.export(exports, "reverseSnippet", ()=>_reverseSnippetDefault.default
);
parcelHelpers.export(exports, "highlight", ()=>_highlightDefault.default
);
parcelHelpers.export(exports, "snippet", ()=>_snippetDefault.default
);
parcelHelpers.export(exports, "insights", ()=>_insightsDefault.default
);
parcelHelpers.export(exports, "getInsightsAnonymousUserToken", ()=>_getInsightsAnonymousUserTokenDefault.default
);
parcelHelpers.export(exports, "getInsightsAnonymousUserTokenInternal", ()=>_getInsightsAnonymousUserToken.getInsightsAnonymousUserTokenInternal
);
var _highlight = require("./highlight");
parcelHelpers.exportAll(_highlight, exports);
var _reverseHighlight = require("./reverseHighlight");
parcelHelpers.exportAll(_reverseHighlight, exports);
var _snippet = require("./snippet");
parcelHelpers.exportAll(_snippet, exports);
var _reverseSnippet = require("./reverseSnippet");
parcelHelpers.exportAll(_reverseSnippet, exports);
var _reverseHighlightDefault = parcelHelpers.interopDefault(_reverseHighlight);
var _reverseSnippetDefault = parcelHelpers.interopDefault(_reverseSnippet);
var _highlightDefault = parcelHelpers.interopDefault(_highlight);
var _snippetDefault = parcelHelpers.interopDefault(_snippet);
var _insights = require("./insights");
var _insightsDefault = parcelHelpers.interopDefault(_insights);
var _getInsightsAnonymousUserToken = require("./get-insights-anonymous-user-token");
var _getInsightsAnonymousUserTokenDefault = parcelHelpers.interopDefault(_getInsightsAnonymousUserToken);

},{"./highlight":"juTzj","./reverseHighlight":"258GF","./snippet":"lMCRi","./reverseSnippet":"5cYoY","./insights":"2EZr9","./get-insights-anonymous-user-token":"cRBQf","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"juTzj":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _suit = require("../lib/suit");
var _utils = require("../lib/utils");
var suit = _suit.component('Highlight');
function highlight(_ref) {
    var attribute = _ref.attribute, _ref$highlightedTagNa = _ref.highlightedTagName, highlightedTagName = _ref$highlightedTagNa === void 0 ? 'mark' : _ref$highlightedTagNa, hit = _ref.hit, _ref$cssClasses = _ref.cssClasses, cssClasses = _ref$cssClasses === void 0 ? {} : _ref$cssClasses;
    var _ref2 = _utils.getPropertyByPath(hit._highlightResult, attribute) || {}, _ref2$value = _ref2.value, attributeValue = _ref2$value === void 0 ? '' : _ref2$value; // cx is not used, since it would be bundled as a dependency for Vue & Angular
    var className = suit({
        descendantName: 'highlighted'
    }) + (cssClasses.highlighted ? " ".concat(cssClasses.highlighted) : '');
    return attributeValue.replace(new RegExp(_utils.TAG_REPLACEMENT.highlightPreTag, 'g'), "<".concat(highlightedTagName, " class=\"").concat(className, "\">")).replace(new RegExp(_utils.TAG_REPLACEMENT.highlightPostTag, 'g'), "</".concat(highlightedTagName, ">"));
}
exports.default = highlight;

},{"../lib/suit":"du81D","../lib/utils":"etVYs","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"du81D":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "component", ()=>component
);
var NAMESPACE = 'ais';
var component = function component(componentName) {
    return function() {
        var _ref = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : {}, descendantName = _ref.descendantName, modifierName = _ref.modifierName;
        var descendent = descendantName ? "-".concat(descendantName) : '';
        var modifier = modifierName ? "--".concat(modifierName) : '';
        return "".concat(NAMESPACE, "-").concat(componentName).concat(descendent).concat(modifier);
    };
};

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"258GF":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _utils = require("../lib/utils");
var _suit = require("../lib/suit");
var suit = _suit.component('ReverseHighlight');
function reverseHighlight(_ref) {
    var attribute = _ref.attribute, _ref$highlightedTagNa = _ref.highlightedTagName, highlightedTagName = _ref$highlightedTagNa === void 0 ? 'mark' : _ref$highlightedTagNa, hit = _ref.hit, _ref$cssClasses = _ref.cssClasses, cssClasses = _ref$cssClasses === void 0 ? {} : _ref$cssClasses;
    var _ref2 = _utils.getPropertyByPath(hit._highlightResult, attribute) || {}, _ref2$value = _ref2.value, attributeValue = _ref2$value === void 0 ? '' : _ref2$value; // cx is not used, since it would be bundled as a dependency for Vue & Angular
    var className = suit({
        descendantName: 'highlighted'
    }) + (cssClasses.highlighted ? " ".concat(cssClasses.highlighted) : '');
    var reverseHighlightedValue = _utils.concatHighlightedParts(_utils.reverseHighlightedParts(_utils.getHighlightedParts(attributeValue)));
    return reverseHighlightedValue.replace(new RegExp(_utils.TAG_REPLACEMENT.highlightPreTag, 'g'), "<".concat(highlightedTagName, " class=\"").concat(className, "\">")).replace(new RegExp(_utils.TAG_REPLACEMENT.highlightPostTag, 'g'), "</".concat(highlightedTagName, ">"));
}
exports.default = reverseHighlight;

},{"../lib/utils":"etVYs","../lib/suit":"du81D","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"lMCRi":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _suit = require("../lib/suit");
var _utils = require("../lib/utils");
var suit = _suit.component('Snippet');
function snippet(_ref) {
    var attribute = _ref.attribute, _ref$highlightedTagNa = _ref.highlightedTagName, highlightedTagName = _ref$highlightedTagNa === void 0 ? 'mark' : _ref$highlightedTagNa, hit = _ref.hit, _ref$cssClasses = _ref.cssClasses, cssClasses = _ref$cssClasses === void 0 ? {} : _ref$cssClasses;
    var _ref2 = _utils.getPropertyByPath(hit._snippetResult, attribute) || {}, _ref2$value = _ref2.value, attributeValue = _ref2$value === void 0 ? '' : _ref2$value; // cx is not used, since it would be bundled as a dependency for Vue & Angular
    var className = suit({
        descendantName: 'highlighted'
    }) + (cssClasses.highlighted ? " ".concat(cssClasses.highlighted) : '');
    return attributeValue.replace(new RegExp(_utils.TAG_REPLACEMENT.highlightPreTag, 'g'), "<".concat(highlightedTagName, " class=\"").concat(className, "\">")).replace(new RegExp(_utils.TAG_REPLACEMENT.highlightPostTag, 'g'), "</".concat(highlightedTagName, ">"));
}
exports.default = snippet;

},{"../lib/suit":"du81D","../lib/utils":"etVYs","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"5cYoY":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _utils = require("../lib/utils");
var _suit = require("../lib/suit");
var suit = _suit.component('ReverseSnippet');
function reverseSnippet(_ref) {
    var attribute = _ref.attribute, _ref$highlightedTagNa = _ref.highlightedTagName, highlightedTagName = _ref$highlightedTagNa === void 0 ? 'mark' : _ref$highlightedTagNa, hit = _ref.hit, _ref$cssClasses = _ref.cssClasses, cssClasses = _ref$cssClasses === void 0 ? {} : _ref$cssClasses;
    var _ref2 = _utils.getPropertyByPath(hit._snippetResult, attribute) || {}, _ref2$value = _ref2.value, attributeValue = _ref2$value === void 0 ? '' : _ref2$value; // cx is not used, since it would be bundled as a dependency for Vue & Angular
    var className = suit({
        descendantName: 'highlighted'
    }) + (cssClasses.highlighted ? " ".concat(cssClasses.highlighted) : '');
    var reverseHighlightedValue = _utils.concatHighlightedParts(_utils.reverseHighlightedParts(_utils.getHighlightedParts(attributeValue)));
    return reverseHighlightedValue.replace(new RegExp(_utils.TAG_REPLACEMENT.highlightPreTag, 'g'), "<".concat(highlightedTagName, " class=\"").concat(className, "\">")).replace(new RegExp(_utils.TAG_REPLACEMENT.highlightPostTag, 'g'), "</".concat(highlightedTagName, ">"));
}
exports.default = reverseSnippet;

},{"../lib/utils":"etVYs","../lib/suit":"du81D","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"2EZr9":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "readDataAttributes", ()=>readDataAttributes
);
parcelHelpers.export(exports, "hasDataAttributes", ()=>hasDataAttributes
);
parcelHelpers.export(exports, "writeDataAttributes", ()=>writeDataAttributes
);
var _utils = require("../lib/utils");
function _typeof(obj1) {
    if (typeof Symbol === "function" && typeof Symbol.iterator === "symbol") _typeof = function _typeof(obj) {
        return typeof obj;
    };
    else _typeof = function _typeof(obj) {
        return obj && typeof Symbol === "function" && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj;
    };
    return _typeof(obj1);
}
function readDataAttributes(domElement) {
    var method = domElement.getAttribute('data-insights-method');
    var serializedPayload = domElement.getAttribute('data-insights-payload');
    if (typeof serializedPayload !== 'string') throw new Error('The insights helper expects `data-insights-payload` to be a base64-encoded JSON string.');
    try {
        var payload = _utils.deserializePayload(serializedPayload);
        return {
            method: method,
            payload: payload
        };
    } catch (error) {
        throw new Error('The insights helper was unable to parse `data-insights-payload`.');
    }
}
function hasDataAttributes(domElement) {
    return domElement.hasAttribute('data-insights-method');
}
function writeDataAttributes(_ref) {
    var method = _ref.method, payload = _ref.payload;
    if (_typeof(payload) !== 'object') throw new Error("The insights helper expects the payload to be an object.");
    var serializedPayload;
    try {
        serializedPayload = _utils.serializePayload(payload);
    } catch (error) {
        throw new Error("Could not JSON serialize the payload object.");
    }
    return "data-insights-method=\"".concat(method, "\" data-insights-payload=\"").concat(serializedPayload, "\"");
}
function insights(method, payload) {
    _utils.warning(false, "`insights` function has been deprecated. It is still supported in 4.x releases, but not further. It is replaced by the `insights` middleware.\n\nFor more information, visit https://www.algolia.com/doc/guides/getting-insights-and-analytics/search-analytics/click-through-and-conversions/how-to/send-click-and-conversion-events-with-instantsearch/js/");
    return writeDataAttributes({
        method: method,
        payload: payload
    });
}
exports.default = insights;

},{"../lib/utils":"etVYs","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"cRBQf":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "ANONYMOUS_TOKEN_COOKIE_KEY", ()=>ANONYMOUS_TOKEN_COOKIE_KEY
);
parcelHelpers.export(exports, "getInsightsAnonymousUserTokenInternal", ()=>getInsightsAnonymousUserTokenInternal
);
var _utils = require("../lib/utils");
var ANONYMOUS_TOKEN_COOKIE_KEY = '_ALGOLIA';
function getCookie(name) {
    var prefix = "".concat(name, "=");
    var cookies = document.cookie.split(';');
    for(var i = 0; i < cookies.length; i++){
        var cookie = cookies[i];
        while(cookie.charAt(0) === ' ')cookie = cookie.substring(1);
        if (cookie.indexOf(prefix) === 0) return cookie.substring(prefix.length, cookie.length);
    }
    return undefined;
}
function getInsightsAnonymousUserTokenInternal() {
    return getCookie(ANONYMOUS_TOKEN_COOKIE_KEY);
}
function getInsightsAnonymousUserToken() {
    _utils.warning(false, "`getInsightsAnonymousUserToken` function has been deprecated. It is still supported in 4.x releases, but not further. It is replaced by the `insights` middleware.\n\nFor more information, visit https://www.algolia.com/doc/guides/getting-insights-and-analytics/search-analytics/click-through-and-conversions/how-to/send-click-and-conversion-events-with-instantsearch/js/");
    return getInsightsAnonymousUserTokenInternal();
}
exports.default = getInsightsAnonymousUserToken;

},{"../lib/utils":"etVYs","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"4mKEu":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "createRouterMiddleware", ()=>createRouterMiddleware
);
var _simple = require("../lib/stateMappings/simple");
var _simpleDefault = parcelHelpers.interopDefault(_simple);
var _history = require("../lib/routers/history");
var _historyDefault = parcelHelpers.interopDefault(_history);
var _utils = require("../lib/utils");
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        if (enumerableOnly) symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        });
        keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = arguments[i] != null ? arguments[i] : {};
        if (i % 2) ownKeys(Object(source), true).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        });
        else if (Object.getOwnPropertyDescriptors) Object.defineProperties(target, Object.getOwnPropertyDescriptors(source));
        else ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
var createRouterMiddleware = function createRouterMiddleware() {
    var props = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : {};
    var _props$router = props.router, router = _props$router === void 0 ? _historyDefault.default() : _props$router, _props$stateMapping = props.stateMapping, stateMapping = _props$stateMapping === void 0 ? _simpleDefault.default() : _props$stateMapping;
    return function(_ref) {
        var instantSearchInstance = _ref.instantSearchInstance;
        function topLevelCreateURL(nextState) {
            var uiState = Object.keys(nextState).reduce(function(acc, indexId) {
                return _objectSpread(_objectSpread({}, acc), {}, _defineProperty({}, indexId, nextState[indexId]));
            }, instantSearchInstance.mainIndex.getWidgetUiState({}));
            var route = stateMapping.stateToRoute(uiState);
            return router.createURL(route);
        }
        instantSearchInstance._createURL = topLevelCreateURL;
        instantSearchInstance._initialUiState = _objectSpread(_objectSpread({}, instantSearchInstance._initialUiState), stateMapping.routeToState(router.read()));
        var lastRouteState = undefined;
        return {
            onStateChange: function onStateChange(_ref2) {
                var uiState = _ref2.uiState;
                var routeState = stateMapping.stateToRoute(uiState);
                if (lastRouteState === undefined || !_utils.isEqual(lastRouteState, routeState)) {
                    router.write(routeState);
                    lastRouteState = routeState;
                }
            },
            subscribe: function subscribe() {
                router.onUpdate(function(route) {
                    instantSearchInstance.setUiState(stateMapping.routeToState(route));
                });
            },
            unsubscribe: function unsubscribe() {
                router.dispose();
            }
        };
    };
};

},{"../lib/stateMappings/simple":"7Ci0f","../lib/routers/history":"haLSt","../lib/utils":"etVYs","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"7Ci0f":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        if (enumerableOnly) symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        });
        keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = arguments[i] != null ? arguments[i] : {};
        if (i % 2) ownKeys(Object(source), true).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        });
        else if (Object.getOwnPropertyDescriptors) Object.defineProperties(target, Object.getOwnPropertyDescriptors(source));
        else ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
function _objectWithoutProperties(source, excluded) {
    if (source == null) return {};
    var target = _objectWithoutPropertiesLoose(source, excluded);
    var key, i;
    if (Object.getOwnPropertySymbols) {
        var sourceSymbolKeys = Object.getOwnPropertySymbols(source);
        for(i = 0; i < sourceSymbolKeys.length; i++){
            key = sourceSymbolKeys[i];
            if (excluded.indexOf(key) >= 0) continue;
            if (!Object.prototype.propertyIsEnumerable.call(source, key)) continue;
            target[key] = source[key];
        }
    }
    return target;
}
function _objectWithoutPropertiesLoose(source, excluded) {
    if (source == null) return {};
    var target = {};
    var sourceKeys = Object.keys(source);
    var key, i;
    for(i = 0; i < sourceKeys.length; i++){
        key = sourceKeys[i];
        if (excluded.indexOf(key) >= 0) continue;
        target[key] = source[key];
    }
    return target;
}
function getIndexStateWithoutConfigure(uiState) {
    var configure = uiState.configure, trackedUiState = _objectWithoutProperties(uiState, [
        "configure"
    ]);
    return trackedUiState;
} // technically a URL could contain any key, since users provide it,
function simpleStateMapping() {
    return {
        stateToRoute: function stateToRoute(uiState) {
            return Object.keys(uiState).reduce(function(state, indexId) {
                return _objectSpread(_objectSpread({}, state), {}, _defineProperty({}, indexId, getIndexStateWithoutConfigure(uiState[indexId])));
            }, {});
        },
        routeToState: function routeToState() {
            var routeState = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : {};
            return Object.keys(routeState).reduce(function(state, indexId) {
                return _objectSpread(_objectSpread({}, state), {}, _defineProperty({}, indexId, getIndexStateWithoutConfigure(routeState[indexId])));
            }, {});
        }
    };
}
exports.default = simpleStateMapping;

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"haLSt":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _qs = require("qs");
var _qsDefault = parcelHelpers.interopDefault(_qs);
function _classCallCheck(instance, Constructor) {
    if (!(instance instanceof Constructor)) throw new TypeError("Cannot call a class as a function");
}
function _defineProperties(target, props) {
    for(var i = 0; i < props.length; i++){
        var descriptor = props[i];
        descriptor.enumerable = descriptor.enumerable || false;
        descriptor.configurable = true;
        if ("value" in descriptor) descriptor.writable = true;
        Object.defineProperty(target, descriptor.key, descriptor);
    }
}
function _createClass(Constructor, protoProps, staticProps) {
    if (protoProps) _defineProperties(Constructor.prototype, protoProps);
    if (staticProps) _defineProperties(Constructor, staticProps);
    return Constructor;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
var defaultCreateURL = function defaultCreateURL(_ref) {
    var qsModule = _ref.qsModule, routeState = _ref.routeState, location = _ref.location;
    var protocol = location.protocol, hostname = location.hostname, _location$port = location.port, port = _location$port === void 0 ? '' : _location$port, pathname = location.pathname, hash = location.hash;
    var queryString = qsModule.stringify(routeState);
    var portWithPrefix = port === '' ? '' : ":".concat(port); // IE <= 11 has no proper `location.origin` so we cannot rely on it.
    if (!queryString) return "".concat(protocol, "//").concat(hostname).concat(portWithPrefix).concat(pathname).concat(hash);
    return "".concat(protocol, "//").concat(hostname).concat(portWithPrefix).concat(pathname, "?").concat(queryString).concat(hash);
};
var defaultParseURL = function defaultParseURL(_ref2) {
    var qsModule = _ref2.qsModule, location = _ref2.location;
    // `qs` by default converts arrays with more than 20 items to an object.
    // We want to avoid this because the data structure manipulated can therefore vary.
    // Setting the limit to `100` seems a good number because the engine's default is 100
    // (it can go up to 1000 but it is very unlikely to select more than 100 items in the UI).
    //
    // Using an `arrayLimit` of `n` allows `n + 1` items.
    //
    // See:
    //   - https://github.com/ljharb/qs#parsing-arrays
    //   - https://www.algolia.com/doc/api-reference/api-parameters/maxValuesPerFacet/
    return qsModule.parse(location.search.slice(1), {
        arrayLimit: 99
    });
};
var setWindowTitle = function setWindowTitle(title) {
    if (title) window.document.title = title;
};
var BrowserHistory = /*#__PURE__*/ function() {
    /**
   * Initializes a new storage provider that syncs the search state to the URL
   * using web APIs (`window.location.pushState` and `onpopstate` event).
   */ function BrowserHistory1() {
        var _ref3 = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : {}, windowTitle = _ref3.windowTitle, _ref3$writeDelay = _ref3.writeDelay, writeDelay = _ref3$writeDelay === void 0 ? 400 : _ref3$writeDelay, _ref3$createURL = _ref3.createURL, createURL = _ref3$createURL === void 0 ? defaultCreateURL : _ref3$createURL, _ref3$parseURL = _ref3.parseURL, parseURL = _ref3$parseURL === void 0 ? defaultParseURL : _ref3$parseURL;
        _classCallCheck(this, BrowserHistory1);
        _defineProperty(this, "windowTitle", void 0);
        _defineProperty(this, "writeDelay", void 0);
        _defineProperty(this, "_createURL", void 0);
        _defineProperty(this, "parseURL", void 0);
        _defineProperty(this, "writeTimer", void 0);
        this.windowTitle = windowTitle;
        this.writeTimer = undefined;
        this.writeDelay = writeDelay;
        this._createURL = createURL;
        this.parseURL = parseURL;
        var title = this.windowTitle && this.windowTitle(this.read());
        setWindowTitle(title);
    }
    /**
   * Reads the URL and returns a syncable UI search state.
   */ _createClass(BrowserHistory1, [
        {
            key: "read",
            value: function read() {
                return this.parseURL({
                    qsModule: _qsDefault.default,
                    location: window.location
                });
            }
        },
        {
            key: "write",
            value: function write(routeState) {
                var _this = this;
                var url = this.createURL(routeState);
                var title = this.windowTitle && this.windowTitle(routeState);
                if (this.writeTimer) window.clearTimeout(this.writeTimer);
                this.writeTimer = window.setTimeout(function() {
                    setWindowTitle(title);
                    window.history.pushState(routeState, title || '', url);
                    _this.writeTimer = undefined;
                }, this.writeDelay);
            }
        },
        {
            key: "onUpdate",
            value: function onUpdate(callback) {
                var _this2 = this;
                this._onPopState = function(event) {
                    if (_this2.writeTimer) {
                        window.clearTimeout(_this2.writeTimer);
                        _this2.writeTimer = undefined;
                    }
                    var routeState = event.state; // At initial load, the state is read from the URL without update.
                    // Therefore the state object is not available.
                    // In this case, we fallback and read the URL.
                    if (!routeState) callback(_this2.read());
                    else callback(routeState);
                };
                window.addEventListener('popstate', this._onPopState);
            }
        },
        {
            key: "createURL",
            value: function createURL(routeState) {
                return this._createURL({
                    qsModule: _qsDefault.default,
                    routeState: routeState,
                    location: window.location
                });
            }
        },
        {
            key: "dispose",
            value: function dispose() {
                if (this._onPopState) window.removeEventListener('popstate', this._onPopState);
                if (this.writeTimer) window.clearTimeout(this.writeTimer);
                this.write({});
            }
        }
    ]);
    return BrowserHistory1;
}();
exports.default = function(props) {
    return new BrowserHistory(props);
};

},{"qs":"kW4GH","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"kW4GH":[function(require,module,exports) {
'use strict';
var stringify = require('./stringify');
var parse = require('./parse');
var formats = require('./formats');
module.exports = {
    formats: formats,
    parse: parse,
    stringify: stringify
};

},{"./stringify":"aJuQi","./parse":"fSZqi","./formats":"d7Ogf"}],"aJuQi":[function(require,module,exports) {
'use strict';
var utils = require('./utils');
var formats = require('./formats');
var has = Object.prototype.hasOwnProperty;
var arrayPrefixGenerators = {
    brackets: function brackets(prefix) {
        return prefix + '[]';
    },
    comma: 'comma',
    indices: function indices(prefix, key) {
        return prefix + '[' + key + ']';
    },
    repeat: function repeat(prefix) {
        return prefix;
    }
};
var isArray = Array.isArray;
var split = String.prototype.split;
var push = Array.prototype.push;
var pushToArray = function(arr, valueOrArray) {
    push.apply(arr, isArray(valueOrArray) ? valueOrArray : [
        valueOrArray
    ]);
};
var toISO = Date.prototype.toISOString;
var defaultFormat = formats['default'];
var defaults = {
    addQueryPrefix: false,
    allowDots: false,
    charset: 'utf-8',
    charsetSentinel: false,
    delimiter: '&',
    encode: true,
    encoder: utils.encode,
    encodeValuesOnly: false,
    format: defaultFormat,
    formatter: formats.formatters[defaultFormat],
    // deprecated
    indices: false,
    serializeDate: function serializeDate(date) {
        return toISO.call(date);
    },
    skipNulls: false,
    strictNullHandling: false
};
var isNonNullishPrimitive = function isNonNullishPrimitive(v) {
    return typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean' || typeof v === 'symbol' || typeof v === 'bigint';
};
var stringify = function stringify1(object, prefix, generateArrayPrefix, strictNullHandling, skipNulls, encoder, filter, sort, allowDots, serializeDate, format, formatter, encodeValuesOnly, charset) {
    var obj = object;
    if (typeof filter === 'function') obj = filter(prefix, obj);
    else if (obj instanceof Date) obj = serializeDate(obj);
    else if (generateArrayPrefix === 'comma' && isArray(obj)) obj = utils.maybeMap(obj, function(value) {
        if (value instanceof Date) return serializeDate(value);
        return value;
    });
    if (obj === null) {
        if (strictNullHandling) return encoder && !encodeValuesOnly ? encoder(prefix, defaults.encoder, charset, 'key', format) : prefix;
        obj = '';
    }
    if (isNonNullishPrimitive(obj) || utils.isBuffer(obj)) {
        if (encoder) {
            var keyValue = encodeValuesOnly ? prefix : encoder(prefix, defaults.encoder, charset, 'key', format);
            if (generateArrayPrefix === 'comma' && encodeValuesOnly) {
                var valuesArray = split.call(String(obj), ',');
                var valuesJoined = '';
                for(var i = 0; i < valuesArray.length; ++i)valuesJoined += (i === 0 ? '' : ',') + formatter(encoder(valuesArray[i], defaults.encoder, charset, 'value', format));
                return [
                    formatter(keyValue) + '=' + valuesJoined
                ];
            }
            return [
                formatter(keyValue) + '=' + formatter(encoder(obj, defaults.encoder, charset, 'value', format))
            ];
        }
        return [
            formatter(prefix) + '=' + formatter(String(obj))
        ];
    }
    var values = [];
    if (typeof obj === 'undefined') return values;
    var objKeys;
    if (generateArrayPrefix === 'comma' && isArray(obj)) // we need to join elements in
    objKeys = [
        {
            value: obj.length > 0 ? obj.join(',') || null : void 0
        }
    ];
    else if (isArray(filter)) objKeys = filter;
    else {
        var keys = Object.keys(obj);
        objKeys = sort ? keys.sort(sort) : keys;
    }
    for(var j = 0; j < objKeys.length; ++j){
        var key = objKeys[j];
        var value1 = typeof key === 'object' && typeof key.value !== 'undefined' ? key.value : obj[key];
        if (skipNulls && value1 === null) continue;
        var keyPrefix = isArray(obj) ? typeof generateArrayPrefix === 'function' ? generateArrayPrefix(prefix, key) : prefix : prefix + (allowDots ? '.' + key : '[' + key + ']');
        pushToArray(values, stringify1(value1, keyPrefix, generateArrayPrefix, strictNullHandling, skipNulls, encoder, filter, sort, allowDots, serializeDate, format, formatter, encodeValuesOnly, charset));
    }
    return values;
};
var normalizeStringifyOptions = function normalizeStringifyOptions(opts) {
    if (!opts) return defaults;
    if (opts.encoder !== null && typeof opts.encoder !== 'undefined' && typeof opts.encoder !== 'function') throw new TypeError('Encoder has to be a function.');
    var charset = opts.charset || defaults.charset;
    if (typeof opts.charset !== 'undefined' && opts.charset !== 'utf-8' && opts.charset !== 'iso-8859-1') throw new TypeError('The charset option must be either utf-8, iso-8859-1, or undefined');
    var format = formats['default'];
    if (typeof opts.format !== 'undefined') {
        if (!has.call(formats.formatters, opts.format)) throw new TypeError('Unknown format option provided.');
        format = opts.format;
    }
    var formatter = formats.formatters[format];
    var filter = defaults.filter;
    if (typeof opts.filter === 'function' || isArray(opts.filter)) filter = opts.filter;
    return {
        addQueryPrefix: typeof opts.addQueryPrefix === 'boolean' ? opts.addQueryPrefix : defaults.addQueryPrefix,
        allowDots: typeof opts.allowDots === 'undefined' ? defaults.allowDots : !!opts.allowDots,
        charset: charset,
        charsetSentinel: typeof opts.charsetSentinel === 'boolean' ? opts.charsetSentinel : defaults.charsetSentinel,
        delimiter: typeof opts.delimiter === 'undefined' ? defaults.delimiter : opts.delimiter,
        encode: typeof opts.encode === 'boolean' ? opts.encode : defaults.encode,
        encoder: typeof opts.encoder === 'function' ? opts.encoder : defaults.encoder,
        encodeValuesOnly: typeof opts.encodeValuesOnly === 'boolean' ? opts.encodeValuesOnly : defaults.encodeValuesOnly,
        filter: filter,
        format: format,
        formatter: formatter,
        serializeDate: typeof opts.serializeDate === 'function' ? opts.serializeDate : defaults.serializeDate,
        skipNulls: typeof opts.skipNulls === 'boolean' ? opts.skipNulls : defaults.skipNulls,
        sort: typeof opts.sort === 'function' ? opts.sort : null,
        strictNullHandling: typeof opts.strictNullHandling === 'boolean' ? opts.strictNullHandling : defaults.strictNullHandling
    };
};
module.exports = function(object, opts) {
    var obj = object;
    var options = normalizeStringifyOptions(opts);
    var objKeys;
    var filter;
    if (typeof options.filter === 'function') {
        filter = options.filter;
        obj = filter('', obj);
    } else if (isArray(options.filter)) {
        filter = options.filter;
        objKeys = filter;
    }
    var keys = [];
    if (typeof obj !== 'object' || obj === null) return '';
    var arrayFormat;
    if (opts && opts.arrayFormat in arrayPrefixGenerators) arrayFormat = opts.arrayFormat;
    else if (opts && 'indices' in opts) arrayFormat = opts.indices ? 'indices' : 'repeat';
    else arrayFormat = 'indices';
    var generateArrayPrefix = arrayPrefixGenerators[arrayFormat];
    if (!objKeys) objKeys = Object.keys(obj);
    if (options.sort) objKeys.sort(options.sort);
    for(var i = 0; i < objKeys.length; ++i){
        var key = objKeys[i];
        if (options.skipNulls && obj[key] === null) continue;
        pushToArray(keys, stringify(obj[key], key, generateArrayPrefix, options.strictNullHandling, options.skipNulls, options.encode ? options.encoder : null, options.filter, options.sort, options.allowDots, options.serializeDate, options.format, options.formatter, options.encodeValuesOnly, options.charset));
    }
    var joined = keys.join(options.delimiter);
    var prefix = options.addQueryPrefix === true ? '?' : '';
    if (options.charsetSentinel) {
        if (options.charset === 'iso-8859-1') // encodeURIComponent('&#10003;'), the "numeric entity" representation of a checkmark
        prefix += 'utf8=%26%2310003%3B&';
        else // encodeURIComponent('✓')
        prefix += 'utf8=%E2%9C%93&';
    }
    return joined.length > 0 ? prefix + joined : '';
};

},{"./utils":"chmkc","./formats":"d7Ogf"}],"chmkc":[function(require,module,exports) {
'use strict';
var formats = require('./formats');
var has = Object.prototype.hasOwnProperty;
var isArray = Array.isArray;
var hexTable = function() {
    var array = [];
    for(var i = 0; i < 256; ++i)array.push('%' + ((i < 16 ? '0' : '') + i.toString(16)).toUpperCase());
    return array;
}();
var compactQueue = function compactQueue(queue) {
    while(queue.length > 1){
        var item = queue.pop();
        var obj = item.obj[item.prop];
        if (isArray(obj)) {
            var compacted = [];
            for(var j = 0; j < obj.length; ++j)if (typeof obj[j] !== 'undefined') compacted.push(obj[j]);
            item.obj[item.prop] = compacted;
        }
    }
};
var arrayToObject = function arrayToObject(source, options) {
    var obj = options && options.plainObjects ? Object.create(null) : {};
    for(var i = 0; i < source.length; ++i)if (typeof source[i] !== 'undefined') obj[i] = source[i];
    return obj;
};
var merge = function merge1(target, source, options) {
    /* eslint no-param-reassign: 0 */ if (!source) return target;
    if (typeof source !== 'object') {
        if (isArray(target)) target.push(source);
        else if (target && typeof target === 'object') {
            if (options && (options.plainObjects || options.allowPrototypes) || !has.call(Object.prototype, source)) target[source] = true;
        } else return [
            target,
            source
        ];
        return target;
    }
    if (!target || typeof target !== 'object') return [
        target
    ].concat(source);
    var mergeTarget = target;
    if (isArray(target) && !isArray(source)) mergeTarget = arrayToObject(target, options);
    if (isArray(target) && isArray(source)) {
        source.forEach(function(item, i) {
            if (has.call(target, i)) {
                var targetItem = target[i];
                if (targetItem && typeof targetItem === 'object' && item && typeof item === 'object') target[i] = merge1(targetItem, item, options);
                else target.push(item);
            } else target[i] = item;
        });
        return target;
    }
    return Object.keys(source).reduce(function(acc, key) {
        var value = source[key];
        if (has.call(acc, key)) acc[key] = merge1(acc[key], value, options);
        else acc[key] = value;
        return acc;
    }, mergeTarget);
};
var assign = function assignSingleSource(target, source) {
    return Object.keys(source).reduce(function(acc, key) {
        acc[key] = source[key];
        return acc;
    }, target);
};
var decode = function(str, decoder, charset) {
    var strWithoutPlus = str.replace(/\+/g, ' ');
    if (charset === 'iso-8859-1') // unescape never throws, no try...catch needed:
    return strWithoutPlus.replace(/%[0-9a-f]{2}/gi, unescape);
    // utf-8
    try {
        return decodeURIComponent(strWithoutPlus);
    } catch (e) {
        return strWithoutPlus;
    }
};
var encode = function encode(str, defaultEncoder, charset, kind, format) {
    // This code was originally written by Brian White (mscdex) for the io.js core querystring library.
    // It has been adapted here for stricter adherence to RFC 3986
    if (str.length === 0) return str;
    var string = str;
    if (typeof str === 'symbol') string = Symbol.prototype.toString.call(str);
    else if (typeof str !== 'string') string = String(str);
    if (charset === 'iso-8859-1') return escape(string).replace(/%u[0-9a-f]{4}/gi, function($0) {
        return '%26%23' + parseInt($0.slice(2), 16) + '%3B';
    });
    var out = '';
    for(var i = 0; i < string.length; ++i){
        var c = string.charCodeAt(i);
        if (c === 0x2D // -
         || c === 0x2E // .
         || c === 0x5F // _
         || c === 0x7E // ~
         || c >= 0x30 && c <= 0x39 // 0-9
         || c >= 0x41 && c <= 0x5A // a-z
         || c >= 0x61 && c <= 0x7A // A-Z
         || format === formats.RFC1738 && (c === 0x28 || c === 0x29 // ( )
        )) {
            out += string.charAt(i);
            continue;
        }
        if (c < 0x80) {
            out = out + hexTable[c];
            continue;
        }
        if (c < 0x800) {
            out = out + (hexTable[0xC0 | c >> 6] + hexTable[0x80 | c & 0x3F]);
            continue;
        }
        if (c < 0xD800 || c >= 0xE000) {
            out = out + (hexTable[0xE0 | c >> 12] + hexTable[0x80 | c >> 6 & 0x3F] + hexTable[0x80 | c & 0x3F]);
            continue;
        }
        i += 1;
        c = 0x10000 + ((c & 0x3FF) << 10 | string.charCodeAt(i) & 0x3FF);
        /* eslint operator-linebreak: [2, "before"] */ out += hexTable[0xF0 | c >> 18] + hexTable[0x80 | c >> 12 & 0x3F] + hexTable[0x80 | c >> 6 & 0x3F] + hexTable[0x80 | c & 0x3F];
    }
    return out;
};
var compact = function compact(value) {
    var queue = [
        {
            obj: {
                o: value
            },
            prop: 'o'
        }
    ];
    var refs = [];
    for(var i = 0; i < queue.length; ++i){
        var item = queue[i];
        var obj = item.obj[item.prop];
        var keys = Object.keys(obj);
        for(var j = 0; j < keys.length; ++j){
            var key = keys[j];
            var val = obj[key];
            if (typeof val === 'object' && val !== null && refs.indexOf(val) === -1) {
                queue.push({
                    obj: obj,
                    prop: key
                });
                refs.push(val);
            }
        }
    }
    compactQueue(queue);
    return value;
};
var isRegExp = function isRegExp(obj) {
    return Object.prototype.toString.call(obj) === '[object RegExp]';
};
var isBuffer = function isBuffer(obj) {
    if (!obj || typeof obj !== 'object') return false;
    return !!(obj.constructor && obj.constructor.isBuffer && obj.constructor.isBuffer(obj));
};
var combine = function combine(a, b) {
    return [].concat(a, b);
};
var maybeMap = function maybeMap(val, fn) {
    if (isArray(val)) {
        var mapped = [];
        for(var i = 0; i < val.length; i += 1)mapped.push(fn(val[i]));
        return mapped;
    }
    return fn(val);
};
module.exports = {
    arrayToObject: arrayToObject,
    assign: assign,
    combine: combine,
    compact: compact,
    decode: decode,
    encode: encode,
    isBuffer: isBuffer,
    isRegExp: isRegExp,
    maybeMap: maybeMap,
    merge: merge
};

},{"./formats":"d7Ogf"}],"d7Ogf":[function(require,module,exports) {
'use strict';
var replace = String.prototype.replace;
var percentTwenties = /%20/g;
var Format = {
    RFC1738: 'RFC1738',
    RFC3986: 'RFC3986'
};
module.exports = {
    'default': Format.RFC3986,
    formatters: {
        RFC1738: function(value) {
            return replace.call(value, percentTwenties, '+');
        },
        RFC3986: function(value) {
            return String(value);
        }
    },
    RFC1738: Format.RFC1738,
    RFC3986: Format.RFC3986
};

},{}],"fSZqi":[function(require,module,exports) {
'use strict';
var utils = require('./utils');
var has = Object.prototype.hasOwnProperty;
var isArray = Array.isArray;
var defaults = {
    allowDots: false,
    allowPrototypes: false,
    arrayLimit: 20,
    charset: 'utf-8',
    charsetSentinel: false,
    comma: false,
    decoder: utils.decode,
    delimiter: '&',
    depth: 5,
    ignoreQueryPrefix: false,
    interpretNumericEntities: false,
    parameterLimit: 1000,
    parseArrays: true,
    plainObjects: false,
    strictNullHandling: false
};
var interpretNumericEntities = function(str) {
    return str.replace(/&#(\d+);/g, function($0, numberStr) {
        return String.fromCharCode(parseInt(numberStr, 10));
    });
};
var parseArrayValue = function(val, options) {
    if (val && typeof val === 'string' && options.comma && val.indexOf(',') > -1) return val.split(',');
    return val;
};
// This is what browsers will submit when the ✓ character occurs in an
// application/x-www-form-urlencoded body and the encoding of the page containing
// the form is iso-8859-1, or when the submitted form has an accept-charset
// attribute of iso-8859-1. Presumably also with other charsets that do not contain
// the ✓ character, such as us-ascii.
var isoSentinel = 'utf8=%26%2310003%3B'; // encodeURIComponent('&#10003;')
// These are the percent-encoded utf-8 octets representing a checkmark, indicating that the request actually is utf-8 encoded.
var charsetSentinel = 'utf8=%E2%9C%93'; // encodeURIComponent('✓')
var parseValues = function parseQueryStringValues(str, options) {
    var obj = {};
    var cleanStr = options.ignoreQueryPrefix ? str.replace(/^\?/, '') : str;
    var limit = options.parameterLimit === Infinity ? undefined : options.parameterLimit;
    var parts = cleanStr.split(options.delimiter, limit);
    var skipIndex = -1; // Keep track of where the utf8 sentinel was found
    var i;
    var charset = options.charset;
    if (options.charsetSentinel) {
        for(i = 0; i < parts.length; ++i)if (parts[i].indexOf('utf8=') === 0) {
            if (parts[i] === charsetSentinel) charset = 'utf-8';
            else if (parts[i] === isoSentinel) charset = 'iso-8859-1';
            skipIndex = i;
            i = parts.length; // The eslint settings do not allow break;
        }
    }
    for(i = 0; i < parts.length; ++i){
        if (i === skipIndex) continue;
        var part = parts[i];
        var bracketEqualsPos = part.indexOf(']=');
        var pos = bracketEqualsPos === -1 ? part.indexOf('=') : bracketEqualsPos + 1;
        var key, val;
        if (pos === -1) {
            key = options.decoder(part, defaults.decoder, charset, 'key');
            val = options.strictNullHandling ? null : '';
        } else {
            key = options.decoder(part.slice(0, pos), defaults.decoder, charset, 'key');
            val = utils.maybeMap(parseArrayValue(part.slice(pos + 1), options), function(encodedVal) {
                return options.decoder(encodedVal, defaults.decoder, charset, 'value');
            });
        }
        if (val && options.interpretNumericEntities && charset === 'iso-8859-1') val = interpretNumericEntities(val);
        if (part.indexOf('[]=') > -1) val = isArray(val) ? [
            val
        ] : val;
        if (has.call(obj, key)) obj[key] = utils.combine(obj[key], val);
        else obj[key] = val;
    }
    return obj;
};
var parseObject = function(chain, val, options, valuesParsed) {
    var leaf = valuesParsed ? val : parseArrayValue(val, options);
    for(var i = chain.length - 1; i >= 0; --i){
        var obj;
        var root = chain[i];
        if (root === '[]' && options.parseArrays) obj = [].concat(leaf);
        else {
            obj = options.plainObjects ? Object.create(null) : {};
            var cleanRoot = root.charAt(0) === '[' && root.charAt(root.length - 1) === ']' ? root.slice(1, -1) : root;
            var index = parseInt(cleanRoot, 10);
            if (!options.parseArrays && cleanRoot === '') obj = {
                0: leaf
            };
            else if (!isNaN(index) && root !== cleanRoot && String(index) === cleanRoot && index >= 0 && options.parseArrays && index <= options.arrayLimit) {
                obj = [];
                obj[index] = leaf;
            } else if (cleanRoot !== '__proto__') obj[cleanRoot] = leaf;
        }
        leaf = obj;
    }
    return leaf;
};
var parseKeys = function parseQueryStringKeys(givenKey, val, options, valuesParsed) {
    if (!givenKey) return;
    // Transform dot notation to bracket notation
    var key = options.allowDots ? givenKey.replace(/\.([^.[]+)/g, '[$1]') : givenKey;
    // The regex chunks
    var brackets = /(\[[^[\]]*])/;
    var child = /(\[[^[\]]*])/g;
    // Get the parent
    var segment = options.depth > 0 && brackets.exec(key);
    var parent = segment ? key.slice(0, segment.index) : key;
    // Stash the parent if it exists
    var keys = [];
    if (parent) {
        // If we aren't using plain objects, optionally prefix keys that would overwrite object prototype properties
        if (!options.plainObjects && has.call(Object.prototype, parent)) {
            if (!options.allowPrototypes) return;
        }
        keys.push(parent);
    }
    // Loop through children appending to the array until we hit depth
    var i = 0;
    while(options.depth > 0 && (segment = child.exec(key)) !== null && i < options.depth){
        i += 1;
        if (!options.plainObjects && has.call(Object.prototype, segment[1].slice(1, -1))) {
            if (!options.allowPrototypes) return;
        }
        keys.push(segment[1]);
    }
    // If there's a remainder, just add whatever is left
    if (segment) keys.push('[' + key.slice(segment.index) + ']');
    return parseObject(keys, val, options, valuesParsed);
};
var normalizeParseOptions = function normalizeParseOptions(opts) {
    if (!opts) return defaults;
    if (opts.decoder !== null && opts.decoder !== undefined && typeof opts.decoder !== 'function') throw new TypeError('Decoder has to be a function.');
    if (typeof opts.charset !== 'undefined' && opts.charset !== 'utf-8' && opts.charset !== 'iso-8859-1') throw new TypeError('The charset option must be either utf-8, iso-8859-1, or undefined');
    var charset = typeof opts.charset === 'undefined' ? defaults.charset : opts.charset;
    return {
        allowDots: typeof opts.allowDots === 'undefined' ? defaults.allowDots : !!opts.allowDots,
        allowPrototypes: typeof opts.allowPrototypes === 'boolean' ? opts.allowPrototypes : defaults.allowPrototypes,
        arrayLimit: typeof opts.arrayLimit === 'number' ? opts.arrayLimit : defaults.arrayLimit,
        charset: charset,
        charsetSentinel: typeof opts.charsetSentinel === 'boolean' ? opts.charsetSentinel : defaults.charsetSentinel,
        comma: typeof opts.comma === 'boolean' ? opts.comma : defaults.comma,
        decoder: typeof opts.decoder === 'function' ? opts.decoder : defaults.decoder,
        delimiter: typeof opts.delimiter === 'string' || utils.isRegExp(opts.delimiter) ? opts.delimiter : defaults.delimiter,
        // eslint-disable-next-line no-implicit-coercion, no-extra-parens
        depth: typeof opts.depth === 'number' || opts.depth === false ? +opts.depth : defaults.depth,
        ignoreQueryPrefix: opts.ignoreQueryPrefix === true,
        interpretNumericEntities: typeof opts.interpretNumericEntities === 'boolean' ? opts.interpretNumericEntities : defaults.interpretNumericEntities,
        parameterLimit: typeof opts.parameterLimit === 'number' ? opts.parameterLimit : defaults.parameterLimit,
        parseArrays: opts.parseArrays !== false,
        plainObjects: typeof opts.plainObjects === 'boolean' ? opts.plainObjects : defaults.plainObjects,
        strictNullHandling: typeof opts.strictNullHandling === 'boolean' ? opts.strictNullHandling : defaults.strictNullHandling
    };
};
module.exports = function(str, opts) {
    var options = normalizeParseOptions(opts);
    if (str === '' || str === null || typeof str === 'undefined') return options.plainObjects ? Object.create(null) : {};
    var tempObj = typeof str === 'string' ? parseValues(str, options) : str;
    var obj = options.plainObjects ? Object.create(null) : {};
    // Iterate over the keys and setup the new object
    var keys = Object.keys(tempObj);
    for(var i = 0; i < keys.length; ++i){
        var key = keys[i];
        var newObj = parseKeys(key, tempObj[key], options, typeof str === 'string');
        obj = utils.merge(obj, newObj, options);
    }
    return utils.compact(obj);
};

},{"./utils":"chmkc"}],"lOLJd":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "isMetadataEnabled", ()=>isMetadataEnabled
);
/**
 * Exposes the metadata of mounted widgets in a custom
 * `<meta name="instantsearch:widgets" />` tag. The metadata per widget is:
 * - applied parameters
 * - widget name
 * - connector name
 */ parcelHelpers.export(exports, "createMetadataMiddleware", ()=>createMetadataMiddleware
);
function _typeof(obj1) {
    if (typeof Symbol === "function" && typeof Symbol.iterator === "symbol") _typeof = function _typeof(obj) {
        return typeof obj;
    };
    else _typeof = function _typeof(obj) {
        return obj && typeof Symbol === "function" && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj;
    };
    return _typeof(obj1);
}
function extractPayload(widgets, instantSearchInstance, payload) {
    var parent = instantSearchInstance.mainIndex;
    var initOptions = {
        instantSearchInstance: instantSearchInstance,
        parent: parent,
        scopedResults: [],
        state: parent.getHelper().state,
        helper: parent.getHelper(),
        createURL: parent.createURL,
        uiState: instantSearchInstance._initialUiState,
        renderState: instantSearchInstance.renderState,
        templatesConfig: instantSearchInstance.templatesConfig,
        searchMetadata: {
            isSearchStalled: instantSearchInstance._isSearchStalled
        }
    };
    widgets.forEach(function(widget) {
        var widgetParams = {};
        if (widget.getWidgetRenderState) {
            var renderState = widget.getWidgetRenderState(initOptions);
            if (renderState && _typeof(renderState.widgetParams) === 'object') widgetParams = renderState.widgetParams;
        } // since we destructure in all widgets, the parameters with defaults are set to "undefined"
        var params = Object.keys(widgetParams).filter(function(key) {
            return widgetParams[key] !== undefined;
        });
        payload.widgets.push({
            type: widget.$$type,
            widgetType: widget.$$widgetType,
            params: params
        });
        if (widget.$$type === 'ais.index') extractPayload(widget.getWidgets(), instantSearchInstance, payload);
    });
}
function isMetadataEnabled() {
    return typeof window !== 'undefined' && window.navigator.userAgent.indexOf('Algolia Crawler') > -1;
}
function createMetadataMiddleware() {
    return function(_ref) {
        var instantSearchInstance = _ref.instantSearchInstance;
        var payload = {
            widgets: []
        };
        var payloadContainer = document.createElement('meta');
        var refNode = document.querySelector('head');
        payloadContainer.name = 'instantsearch:widgets';
        return {
            onStateChange: function onStateChange() {},
            subscribe: function subscribe() {
                // using setTimeout here to delay extraction until widgets have been added in a tick (e.g. Vue)
                setTimeout(function() {
                    var client = instantSearchInstance.client;
                    payload.ua = client.transporter && client.transporter.userAgent ? client.transporter.userAgent.value : client._ua;
                    extractPayload(instantSearchInstance.mainIndex.getWidgets(), instantSearchInstance, payload);
                    payloadContainer.content = JSON.stringify(payload);
                    refNode.appendChild(payloadContainer);
                }, 0);
            },
            unsubscribe: function unsubscribe() {
                payloadContainer.remove();
            }
        };
    };
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"co24K":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "createInfiniteHitsSessionStorageCache", ()=>_sessionStorageDefault.default
);
var _sessionStorage = require("./sessionStorage");
var _sessionStorageDefault = parcelHelpers.interopDefault(_sessionStorage);

},{"./sessionStorage":"gzyTs","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"gzyTs":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _utils = require("../utils");
function _objectWithoutProperties(source, excluded) {
    if (source == null) return {};
    var target = _objectWithoutPropertiesLoose(source, excluded);
    var key, i;
    if (Object.getOwnPropertySymbols) {
        var sourceSymbolKeys = Object.getOwnPropertySymbols(source);
        for(i = 0; i < sourceSymbolKeys.length; i++){
            key = sourceSymbolKeys[i];
            if (excluded.indexOf(key) >= 0) continue;
            if (!Object.prototype.propertyIsEnumerable.call(source, key)) continue;
            target[key] = source[key];
        }
    }
    return target;
}
function _objectWithoutPropertiesLoose(source, excluded) {
    if (source == null) return {};
    var target = {};
    var sourceKeys = Object.keys(source);
    var key, i;
    for(i = 0; i < sourceKeys.length; i++){
        key = sourceKeys[i];
        if (excluded.indexOf(key) >= 0) continue;
        target[key] = source[key];
    }
    return target;
}
function getStateWithoutPage(state) {
    var _ref = state || {}, page = _ref.page, rest = _objectWithoutProperties(_ref, [
        "page"
    ]);
    return rest;
}
var KEY = 'ais.infiniteHits';
function hasSessionStorage() {
    return typeof window !== 'undefined' && typeof window.sessionStorage !== 'undefined';
}
function createInfiniteHitsSessionStorageCache() {
    return {
        read: function read(_ref2) {
            var state = _ref2.state;
            if (!hasSessionStorage()) return null;
            try {
                var cache = JSON.parse(window.sessionStorage.getItem(KEY));
                return cache && _utils.isEqual(cache.state, getStateWithoutPage(state)) ? cache.hits : null;
            } catch (error) {
                if (error instanceof SyntaxError) try {
                    window.sessionStorage.removeItem(KEY);
                } catch (err) {}
                return null;
            }
        },
        write: function write(_ref3) {
            var state = _ref3.state, hits = _ref3.hits;
            if (!hasSessionStorage()) return;
            try {
                window.sessionStorage.setItem(KEY, JSON.stringify({
                    state: getStateWithoutPage(state),
                    hits: hits
                }));
            } catch (error) {}
        }
    };
}
exports.default = createInfiniteHitsSessionStorageCache;

},{"../utils":"etVYs","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"ehDkI":[function(require,module,exports) {
/*! algoliasearch-lite.umd.js | 4.9.1 | © Algolia, inc. | https://github.com/algolia/algoliasearch-client-javascript */ !function(e, t) {
    module.exports = t();
}(this, function() {
    "use strict";
    function e1(e, t, r) {
        return t in e ? Object.defineProperty(e, t, {
            value: r,
            enumerable: !0,
            configurable: !0,
            writable: !0
        }) : e[t] = r, e;
    }
    function t1(e, t2) {
        var r = Object.keys(e);
        if (Object.getOwnPropertySymbols) {
            var n = Object.getOwnPropertySymbols(e);
            t2 && (n = n.filter(function(t) {
                return Object.getOwnPropertyDescriptor(e, t).enumerable;
            })), r.push.apply(r, n);
        }
        return r;
    }
    function r1(r) {
        for(var n = 1; n < arguments.length; n++){
            var o = null != arguments[n] ? arguments[n] : {};
            n % 2 ? t1(Object(o), !0).forEach(function(t) {
                e1(r, t, o[t]);
            }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(r, Object.getOwnPropertyDescriptors(o)) : t1(Object(o)).forEach(function(e) {
                Object.defineProperty(r, e, Object.getOwnPropertyDescriptor(o, e));
            });
        }
        return r;
    }
    function n1(e2, t3) {
        if (null == e2) return {};
        var r2, n2, o2 = function(e, t) {
            if (null == e) return {};
            var r, n, o = {}, a = Object.keys(e);
            for(n = 0; n < a.length; n++)r = a[n], t.indexOf(r) >= 0 || (o[r] = e[r]);
            return o;
        }(e2, t3);
        if (Object.getOwnPropertySymbols) {
            var a2 = Object.getOwnPropertySymbols(e2);
            for(n2 = 0; n2 < a2.length; n2++)r2 = a2[n2], t3.indexOf(r2) >= 0 || Object.prototype.propertyIsEnumerable.call(e2, r2) && (o2[r2] = e2[r2]);
        }
        return o2;
    }
    function o1(e3, t4) {
        return function(e) {
            if (Array.isArray(e)) return e;
        }(e3) || function(e, t) {
            if (!(Symbol.iterator in Object(e) || "[object Arguments]" === Object.prototype.toString.call(e))) return;
            var r = [], n = !0, o = !1, a = void 0;
            try {
                for(var u, i = e[Symbol.iterator](); !(n = (u = i.next()).done) && (r.push(u.value), !t || r.length !== t); n = !0);
            } catch (e4) {
                o = !0, a = e4;
            } finally{
                try {
                    n || null == i.return || i.return();
                } finally{
                    if (o) throw a;
                }
            }
            return r;
        }(e3, t4) || function() {
            throw new TypeError("Invalid attempt to destructure non-iterable instance");
        }();
    }
    function a1(e5) {
        return function(e) {
            if (Array.isArray(e)) {
                for(var t = 0, r = new Array(e.length); t < e.length; t++)r[t] = e[t];
                return r;
            }
        }(e5) || function(e) {
            if (Symbol.iterator in Object(e) || "[object Arguments]" === Object.prototype.toString.call(e)) return Array.from(e);
        }(e5) || function() {
            throw new TypeError("Invalid attempt to spread non-iterable instance");
        }();
    }
    function u1(e6) {
        var t5, r3 = "algoliasearch-client-js-".concat(e6.key), n3 = function() {
            return void 0 === t5 && (t5 = e6.localStorage || window.localStorage), t5;
        }, a3 = function() {
            return JSON.parse(n3().getItem(r3) || "{}");
        };
        return {
            get: function(e7, t6) {
                var r4 = arguments.length > 2 && void 0 !== arguments[2] ? arguments[2] : {
                    miss: function() {
                        return Promise.resolve();
                    }
                };
                return Promise.resolve().then(function() {
                    var r = JSON.stringify(e7), n = a3()[r];
                    return Promise.all([
                        n || t6(),
                        void 0 !== n
                    ]);
                }).then(function(e) {
                    var t = o1(e, 2), n = t[0], a = t[1];
                    return Promise.all([
                        n,
                        a || r4.miss(n)
                    ]);
                }).then(function(e) {
                    return o1(e, 1)[0];
                });
            },
            set: function(e, t) {
                return Promise.resolve().then(function() {
                    var o = a3();
                    return o[JSON.stringify(e)] = t, n3().setItem(r3, JSON.stringify(o)), t;
                });
            },
            delete: function(e) {
                return Promise.resolve().then(function() {
                    var t = a3();
                    delete t[JSON.stringify(e)], n3().setItem(r3, JSON.stringify(t));
                });
            },
            clear: function() {
                return Promise.resolve().then(function() {
                    n3().removeItem(r3);
                });
            }
        };
    }
    function i1(e8) {
        var t7 = a1(e8.caches), r5 = t7.shift();
        return void 0 === r5 ? {
            get: function(e9, t) {
                var r = arguments.length > 2 && void 0 !== arguments[2] ? arguments[2] : {
                    miss: function() {
                        return Promise.resolve();
                    }
                }, n = t();
                return n.then(function(e) {
                    return Promise.all([
                        e,
                        r.miss(e)
                    ]);
                }).then(function(e) {
                    return o1(e, 1)[0];
                });
            },
            set: function(e, t) {
                return Promise.resolve(t);
            },
            delete: function(e) {
                return Promise.resolve();
            },
            clear: function() {
                return Promise.resolve();
            }
        } : {
            get: function(e, n) {
                var o = arguments.length > 2 && void 0 !== arguments[2] ? arguments[2] : {
                    miss: function() {
                        return Promise.resolve();
                    }
                };
                return r5.get(e, n, o).catch(function() {
                    return i1({
                        caches: t7
                    }).get(e, n, o);
                });
            },
            set: function(e, n) {
                return r5.set(e, n).catch(function() {
                    return i1({
                        caches: t7
                    }).set(e, n);
                });
            },
            delete: function(e) {
                return r5.delete(e).catch(function() {
                    return i1({
                        caches: t7
                    }).delete(e);
                });
            },
            clear: function() {
                return r5.clear().catch(function() {
                    return i1({
                        caches: t7
                    }).clear();
                });
            }
        };
    }
    function s1() {
        var e10 = arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : {
            serializable: !0
        }, t = {};
        return {
            get: function(r, n) {
                var o = arguments.length > 2 && void 0 !== arguments[2] ? arguments[2] : {
                    miss: function() {
                        return Promise.resolve();
                    }
                }, a = JSON.stringify(r);
                if (a in t) return Promise.resolve(e10.serializable ? JSON.parse(t[a]) : t[a]);
                var u = n(), i = o && o.miss || function() {
                    return Promise.resolve();
                };
                return u.then(function(e) {
                    return i(e);
                }).then(function() {
                    return u;
                });
            },
            set: function(r, n) {
                return t[JSON.stringify(r)] = e10.serializable ? JSON.stringify(n) : n, Promise.resolve(n);
            },
            delete: function(e) {
                return delete t[JSON.stringify(e)], Promise.resolve();
            },
            clear: function() {
                return t = {}, Promise.resolve();
            }
        };
    }
    function c1(e) {
        for(var t = e.length - 1; t > 0; t--){
            var r = Math.floor(Math.random() * (t + 1)), n = e[t];
            e[t] = e[r], e[r] = n;
        }
        return e;
    }
    function l1(e, t) {
        return t ? (Object.keys(t).forEach(function(r) {
            e[r] = t[r](e);
        }), e) : e;
    }
    function f1(e) {
        for(var t = arguments.length, r = new Array(t > 1 ? t - 1 : 0), n = 1; n < t; n++)r[n - 1] = arguments[n];
        var o = 0;
        return e.replace(/%s/g, function() {
            return encodeURIComponent(r[o++]);
        });
    }
    var h1 = {
        WithinQueryParameters: 0,
        WithinHeaders: 1
    };
    function d1(e11, t) {
        var r = e11 || {}, n = r.data || {};
        return Object.keys(r).forEach(function(e) {
            -1 === [
                "timeout",
                "headers",
                "queryParameters",
                "data",
                "cacheable"
            ].indexOf(e) && (n[e] = r[e]);
        }), {
            data: Object.entries(n).length > 0 ? n : void 0,
            timeout: r.timeout || t,
            headers: r.headers || {},
            queryParameters: r.queryParameters || {},
            cacheable: r.cacheable
        };
    }
    var m1 = {
        Read: 1,
        Write: 2,
        Any: 3
    }, p1 = 1, v = 2, y = 3;
    function g(e) {
        var t = arguments.length > 1 && void 0 !== arguments[1] ? arguments[1] : p1;
        return r1(r1({}, e), {}, {
            status: t,
            lastUpdate: Date.now()
        });
    }
    function b(e) {
        return "string" == typeof e ? {
            protocol: "https",
            url: e,
            accept: m1.Any
        } : {
            protocol: e.protocol || "https",
            url: e.url,
            accept: e.accept || m1.Any
        };
    }
    var O = "GET", P = "POST";
    function q(e12, t8) {
        return Promise.all(t8.map(function(t) {
            return e12.get(t, function() {
                return Promise.resolve(g(t));
            });
        })).then(function(e13) {
            var r = e13.filter(function(e14) {
                return function(e) {
                    return e.status === p1 || Date.now() - e.lastUpdate > 12e4;
                }(e14);
            }), n = e13.filter(function(e15) {
                return function(e) {
                    return e.status === y && Date.now() - e.lastUpdate <= 12e4;
                }(e15);
            }), o = [].concat(a1(r), a1(n));
            return {
                getTimeout: function(e, t) {
                    return (0 === n.length && 0 === e ? 1 : n.length + 3 + e) * t;
                },
                statelessHosts: o.length > 0 ? o.map(function(e) {
                    return b(e);
                }) : t8
            };
        });
    }
    function j(e16, t9, n4, o3) {
        var u = [], i = function(e, t) {
            if (e.method === O || void 0 === e.data && void 0 === t.data) return;
            var n = Array.isArray(e.data) ? e.data : r1(r1({}, e.data), t.data);
            return JSON.stringify(n);
        }(n4, o3), s = function(e17, t10) {
            var n = r1(r1({}, e17.headers), t10.headers), o = {};
            return Object.keys(n).forEach(function(e) {
                var t = n[e];
                o[e.toLowerCase()] = t;
            }), o;
        }(e16, o3), c = n4.method, l2 = n4.method !== O ? {} : r1(r1({}, n4.data), o3.data), f = r1(r1(r1({
            "x-algolia-agent": e16.userAgent.value
        }, e16.queryParameters), l2), o3.queryParameters), h = 0, d2 = function t11(r6, a) {
            var l = r6.pop();
            if (void 0 === l) throw {
                name: "RetryError",
                message: "Unreachable hosts - your application id may be incorrect. If the error persists, contact support@algolia.com.",
                transporterStackTrace: A(u)
            };
            var d = {
                data: i,
                headers: s,
                method: c,
                url: S(l, n4.path, f),
                connectTimeout: a(h, e16.timeouts.connect),
                responseTimeout: a(h, o3.timeout)
            }, m = function(e) {
                var t = {
                    request: d,
                    response: e,
                    host: l,
                    triesLeft: r6.length
                };
                return u.push(t), t;
            }, p = {
                onSuccess: function(e18) {
                    return function(e19) {
                        try {
                            return JSON.parse(e19.content);
                        } catch (t12) {
                            throw function(e, t) {
                                return {
                                    name: "DeserializationError",
                                    message: e,
                                    response: t
                                };
                            }(t12.message, e19);
                        }
                    }(e18);
                },
                onRetry: function(n) {
                    var o = m(n);
                    return n.isTimedOut && h++, Promise.all([
                        e16.logger.info("Retryable failure", x(o)),
                        e16.hostsCache.set(l, g(l, n.isTimedOut ? y : v))
                    ]).then(function() {
                        return t11(r6, a);
                    });
                },
                onFail: function(e20) {
                    throw m(e20), function(e22, t13) {
                        var r7 = e22.content, n = e22.status, o = r7;
                        try {
                            o = JSON.parse(r7).message;
                        } catch (e21) {}
                        return function(e, t, r) {
                            return {
                                name: "ApiError",
                                message: e,
                                status: t,
                                transporterStackTrace: r
                            };
                        }(o, n, t13);
                    }(e20, A(u));
                }
            };
            return e16.requester.send(d).then(function(e23) {
                return function(e24, t14) {
                    return function(e25) {
                        var t15 = e25.status;
                        return e25.isTimedOut || function(e) {
                            var t = e.isTimedOut, r = e.status;
                            return !t && 0 == ~~r;
                        }(e25) || 2 != ~~(t15 / 100) && 4 != ~~(t15 / 100);
                    }(e24) ? t14.onRetry(e24) : 2 == ~~(e24.status / 100) ? t14.onSuccess(e24) : t14.onFail(e24);
                }(e23, p);
            });
        };
        return q(e16.hostsCache, t9).then(function(e) {
            return d2(a1(e.statelessHosts).reverse(), e.getTimeout);
        });
    }
    function w(e26) {
        var t = {
            value: "Algolia for JavaScript (".concat(e26, ")"),
            add: function(e) {
                var r = "; ".concat(e.segment).concat(void 0 !== e.version ? " (".concat(e.version, ")") : "");
                return -1 === t.value.indexOf(r) && (t.value = "".concat(t.value).concat(r)), t;
            }
        };
        return t;
    }
    function S(e, t, r) {
        var n = T(r), o = "".concat(e.protocol, "://").concat(e.url, "/").concat("/" === t.charAt(0) ? t.substr(1) : t);
        return n.length && (o += "?".concat(n)), o;
    }
    function T(e) {
        return Object.keys(e).map(function(t) {
            var r;
            return f1("%s=%s", t, (r = e[t], "[object Object]" === Object.prototype.toString.call(r) || "[object Array]" === Object.prototype.toString.call(r) ? JSON.stringify(e[t]) : e[t]));
        }).join("&");
    }
    function A(e27) {
        return e27.map(function(e) {
            return x(e);
        });
    }
    function x(e) {
        var t = e.request.headers["x-algolia-api-key"] ? {
            "x-algolia-api-key": "*****"
        } : {};
        return r1(r1({}, e), {}, {
            request: r1(r1({}, e.request), {}, {
                headers: r1(r1({}, e.request.headers), t)
            })
        });
    }
    var N = function(e28) {
        var t16 = e28.appId, n5 = function(e, t, r) {
            var n = {
                "x-algolia-api-key": r,
                "x-algolia-application-id": t
            };
            return {
                headers: function() {
                    return e === h1.WithinHeaders ? n : {};
                },
                queryParameters: function() {
                    return e === h1.WithinQueryParameters ? n : {};
                }
            };
        }(void 0 !== e28.authMode ? e28.authMode : h1.WithinHeaders, t16, e28.apiKey), a4 = function(e29) {
            var t17 = e29.hostsCache, r8 = e29.logger, n6 = e29.requester, a5 = e29.requestsCache, u = e29.responsesCache, i = e29.timeouts, s = e29.userAgent, c = e29.hosts, l = e29.queryParameters, f = {
                hostsCache: t17,
                logger: r8,
                requester: n6,
                requestsCache: a5,
                responsesCache: u,
                timeouts: i,
                userAgent: s,
                headers: e29.headers,
                queryParameters: l,
                hosts: c.map(function(e) {
                    return b(e);
                }),
                read: function(e30, t18) {
                    var r = d1(t18, f.timeouts.read), n = function() {
                        return j(f, f.hosts.filter(function(e) {
                            return 0 != (e.accept & m1.Read);
                        }), e30, r);
                    };
                    if (!0 !== (void 0 !== r.cacheable ? r.cacheable : e30.cacheable)) return n();
                    var a = {
                        request: e30,
                        mappedRequestOptions: r,
                        transporter: {
                            queryParameters: f.queryParameters,
                            headers: f.headers
                        }
                    };
                    return f.responsesCache.get(a, function() {
                        return f.requestsCache.get(a, function() {
                            return f.requestsCache.set(a, n()).then(function(e) {
                                return Promise.all([
                                    f.requestsCache.delete(a),
                                    e
                                ]);
                            }, function(e) {
                                return Promise.all([
                                    f.requestsCache.delete(a),
                                    Promise.reject(e)
                                ]);
                            }).then(function(e) {
                                var t = o1(e, 2);
                                t[0];
                                return t[1];
                            });
                        });
                    }, {
                        miss: function(e) {
                            return f.responsesCache.set(a, e);
                        }
                    });
                },
                write: function(e31, t) {
                    return j(f, f.hosts.filter(function(e) {
                        return 0 != (e.accept & m1.Write);
                    }), e31, d1(t, f.timeouts.write));
                }
            };
            return f;
        }(r1(r1({
            hosts: [
                {
                    url: "".concat(t16, "-dsn.algolia.net"),
                    accept: m1.Read
                },
                {
                    url: "".concat(t16, ".algolia.net"),
                    accept: m1.Write
                }
            ].concat(c1([
                {
                    url: "".concat(t16, "-1.algolianet.com")
                },
                {
                    url: "".concat(t16, "-2.algolianet.com")
                },
                {
                    url: "".concat(t16, "-3.algolianet.com")
                }
            ]))
        }, e28), {}, {
            headers: r1(r1(r1({}, n5.headers()), {
                "content-type": "application/x-www-form-urlencoded"
            }), e28.headers),
            queryParameters: r1(r1({}, n5.queryParameters()), e28.queryParameters)
        }));
        return l1({
            transporter: a4,
            appId: t16,
            addAlgoliaAgent: function(e, t) {
                a4.userAgent.add({
                    segment: e,
                    version: t
                });
            },
            clearCache: function() {
                return Promise.all([
                    a4.requestsCache.clear(),
                    a4.responsesCache.clear()
                ]).then(function() {});
            }
        }, e28.methods);
    }, C = function(e) {
        return function(t) {
            var r = arguments.length > 1 && void 0 !== arguments[1] ? arguments[1] : {}, n = {
                transporter: e.transporter,
                appId: e.appId,
                indexName: t
            };
            return l1(n, r.methods);
        };
    }, k = function(e32) {
        return function(t, n) {
            var o = t.map(function(e) {
                return r1(r1({}, e), {}, {
                    params: T(e.params || {})
                });
            });
            return e32.transporter.read({
                method: P,
                path: "1/indexes/*/queries",
                data: {
                    requests: o
                },
                cacheable: !0
            }, n);
        };
    }, J = function(e) {
        return function(t19, o) {
            return Promise.all(t19.map(function(t) {
                var a = t.params, u = a.facetName, i = a.facetQuery, s = n1(a, [
                    "facetName",
                    "facetQuery"
                ]);
                return C(e)(t.indexName, {
                    methods: {
                        searchForFacetValues: F
                    }
                }).searchForFacetValues(u, i, r1(r1({}, o), s));
            }));
        };
    }, E = function(e) {
        return function(t, r, n) {
            return e.transporter.read({
                method: P,
                path: f1("1/answers/%s/prediction", e.indexName),
                data: {
                    query: t,
                    queryLanguages: r
                },
                cacheable: !0
            }, n);
        };
    }, I = function(e) {
        return function(t, r) {
            return e.transporter.read({
                method: P,
                path: f1("1/indexes/%s/query", e.indexName),
                data: {
                    query: t
                },
                cacheable: !0
            }, r);
        };
    }, F = function(e) {
        return function(t, r, n) {
            return e.transporter.read({
                method: P,
                path: f1("1/indexes/%s/facets/%s/query", e.indexName, t),
                data: {
                    facetQuery: r
                },
                cacheable: !0
            }, n);
        };
    }, R = 1, D = 2, W = 3;
    function H(e33, t20, n7) {
        var o4, a6 = {
            appId: e33,
            apiKey: t20,
            timeouts: {
                connect: 1,
                read: 2,
                write: 30
            },
            requester: {
                send: function(e34) {
                    return new Promise(function(t21) {
                        var r = new XMLHttpRequest;
                        r.open(e34.method, e34.url, !0), Object.keys(e34.headers).forEach(function(t) {
                            return r.setRequestHeader(t, e34.headers[t]);
                        });
                        var n8, o = function(e, n) {
                            return setTimeout(function() {
                                r.abort(), t21({
                                    status: 0,
                                    content: n,
                                    isTimedOut: !0
                                });
                            }, 1e3 * e);
                        }, a = o(e34.connectTimeout, "Connection timeout");
                        r.onreadystatechange = function() {
                            r.readyState > r.OPENED && void 0 === n8 && (clearTimeout(a), n8 = o(e34.responseTimeout, "Socket timeout"));
                        }, r.onerror = function() {
                            0 === r.status && (clearTimeout(a), clearTimeout(n8), t21({
                                content: r.responseText || "Network request failed",
                                status: r.status,
                                isTimedOut: !1
                            }));
                        }, r.onload = function() {
                            clearTimeout(a), clearTimeout(n8), t21({
                                content: r.responseText,
                                status: r.status,
                                isTimedOut: !1
                            });
                        }, r.send(e34.data);
                    });
                }
            },
            logger: (o4 = W, {
                debug: function(e, t) {
                    return R >= o4 && console.debug(e, t), Promise.resolve();
                },
                info: function(e, t) {
                    return D >= o4 && console.info(e, t), Promise.resolve();
                },
                error: function(e, t) {
                    return console.error(e, t), Promise.resolve();
                }
            }),
            responsesCache: s1(),
            requestsCache: s1({
                serializable: !1
            }),
            hostsCache: i1({
                caches: [
                    u1({
                        key: "".concat("4.9.1", "-").concat(e33)
                    }),
                    s1()
                ]
            }),
            userAgent: w("4.9.1").add({
                segment: "Browser",
                version: "lite"
            }),
            authMode: h1.WithinQueryParameters
        };
        return N(r1(r1(r1({}, a6), n7), {}, {
            methods: {
                search: k,
                searchForFacetValues: J,
                multipleQueries: k,
                multipleSearchForFacetValues: J,
                initIndex: function(e) {
                    return function(t) {
                        return C(e)(t, {
                            methods: {
                                search: I,
                                searchForFacetValues: F,
                                findAnswers: E
                            }
                        });
                    };
                }
            }
        }));
    }
    return H.version = "4.9.1", H;
});

},{}],"3Syxs":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _autocomplete = require("./autocomplete");
parcelHelpers.exportAll(_autocomplete, exports);
var _requesters = require("./requesters");
parcelHelpers.exportAll(_requesters, exports);
var _types = require("./types");
parcelHelpers.exportAll(_types, exports);

},{"./autocomplete":"8CeYd","./requesters":"24Y2H","./types":"8cK74","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"8CeYd":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "autocomplete", ()=>autocomplete
);
var _autocompleteCore = require("@algolia/autocomplete-core");
var _autocompleteShared = require("@algolia/autocomplete-shared");
var _htm = require("htm");
var _htmDefault = parcelHelpers.interopDefault(_htm);
var _createAutocompleteDom = require("./createAutocompleteDom");
var _createEffectWrapper = require("./createEffectWrapper");
var _createReactiveWrapper = require("./createReactiveWrapper");
var _getDefaultOptions = require("./getDefaultOptions");
var _getPanelPlacementStyle = require("./getPanelPlacementStyle");
var _render = require("./render");
var _userAgents = require("./userAgents");
var _utils = require("./utils");
var _excluded = [
    "components"
];
function _objectWithoutProperties(source, excluded) {
    if (source == null) return {};
    var target = _objectWithoutPropertiesLoose(source, excluded);
    var key, i;
    if (Object.getOwnPropertySymbols) {
        var sourceSymbolKeys = Object.getOwnPropertySymbols(source);
        for(i = 0; i < sourceSymbolKeys.length; i++){
            key = sourceSymbolKeys[i];
            if (excluded.indexOf(key) >= 0) continue;
            if (!Object.prototype.propertyIsEnumerable.call(source, key)) continue;
            target[key] = source[key];
        }
    }
    return target;
}
function _objectWithoutPropertiesLoose(source, excluded) {
    if (source == null) return {};
    var target = {};
    var sourceKeys = Object.keys(source);
    var key, i;
    for(i = 0; i < sourceKeys.length; i++){
        key = sourceKeys[i];
        if (excluded.indexOf(key) >= 0) continue;
        target[key] = source[key];
    }
    return target;
}
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        enumerableOnly && (symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        })), keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = null != arguments[i] ? arguments[i] : {};
        i % 2 ? ownKeys(Object(source), !0).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
function autocomplete(options) {
    var _createEffectWrapper1 = _createEffectWrapper.createEffectWrapper(), runEffect = _createEffectWrapper1.runEffect, cleanupEffects = _createEffectWrapper1.cleanupEffects, runEffects = _createEffectWrapper1.runEffects;
    var _createReactiveWrappe = _createReactiveWrapper.createReactiveWrapper(), reactive = _createReactiveWrappe.reactive, runReactives = _createReactiveWrappe.runReactives;
    var hasNoResultsSourceTemplateRef = _autocompleteShared.createRef(false);
    var optionsRef = _autocompleteShared.createRef(options);
    var onStateChangeRef = _autocompleteShared.createRef(undefined);
    var props = reactive(function() {
        return _getDefaultOptions.getDefaultOptions(optionsRef.current);
    });
    var isDetached = reactive(function() {
        return props.value.core.environment.matchMedia(props.value.renderer.detachedMediaQuery).matches;
    });
    var autocomplete1 = reactive(function() {
        return _autocompleteCore.createAutocomplete(_objectSpread(_objectSpread({}, props.value.core), {}, {
            onStateChange: function onStateChange(params) {
                var _onStateChangeRef$cur, _props$value$core$onS, _props$value$core;
                hasNoResultsSourceTemplateRef.current = params.state.collections.some(function(collection) {
                    return collection.source.templates.noResults;
                });
                (_onStateChangeRef$cur = onStateChangeRef.current) === null || _onStateChangeRef$cur === void 0 || _onStateChangeRef$cur.call(onStateChangeRef, params);
                (_props$value$core$onS = (_props$value$core = props.value.core).onStateChange) === null || _props$value$core$onS === void 0 || _props$value$core$onS.call(_props$value$core, params);
            },
            shouldPanelOpen: optionsRef.current.shouldPanelOpen || function(_ref) {
                var state = _ref.state;
                if (isDetached.value) return true;
                var hasItems = _autocompleteShared.getItemsCount(state) > 0;
                if (!props.value.core.openOnFocus && !state.query) return hasItems;
                var hasNoResultsTemplate = Boolean(hasNoResultsSourceTemplateRef.current || props.value.renderer.renderNoResults);
                return !hasItems && hasNoResultsTemplate || hasItems;
            },
            __autocomplete_metadata: {
                userAgents: _userAgents.userAgents,
                options: options
            }
        }));
    });
    var lastStateRef = _autocompleteShared.createRef(_objectSpread({
        collections: [],
        completion: null,
        context: {},
        isOpen: false,
        query: '',
        activeItemId: null,
        status: 'idle'
    }, props.value.core.initialState));
    var propGetters = {
        getEnvironmentProps: props.value.renderer.getEnvironmentProps,
        getFormProps: props.value.renderer.getFormProps,
        getInputProps: props.value.renderer.getInputProps,
        getItemProps: props.value.renderer.getItemProps,
        getLabelProps: props.value.renderer.getLabelProps,
        getListProps: props.value.renderer.getListProps,
        getPanelProps: props.value.renderer.getPanelProps,
        getRootProps: props.value.renderer.getRootProps
    };
    var autocompleteScopeApi = {
        setActiveItemId: autocomplete1.value.setActiveItemId,
        setQuery: autocomplete1.value.setQuery,
        setCollections: autocomplete1.value.setCollections,
        setIsOpen: autocomplete1.value.setIsOpen,
        setStatus: autocomplete1.value.setStatus,
        setContext: autocomplete1.value.setContext,
        refresh: autocomplete1.value.refresh
    };
    var html = reactive(function() {
        return _htmDefault.default.bind(props.value.renderer.renderer.createElement);
    });
    var dom = reactive(function() {
        return _createAutocompleteDom.createAutocompleteDom({
            autocomplete: autocomplete1.value,
            autocompleteScopeApi: autocompleteScopeApi,
            classNames: props.value.renderer.classNames,
            environment: props.value.core.environment,
            isDetached: isDetached.value,
            placeholder: props.value.core.placeholder,
            propGetters: propGetters,
            setIsModalOpen: setIsModalOpen,
            state: lastStateRef.current,
            translations: props.value.renderer.translations
        });
    });
    function setPanelPosition() {
        _utils.setProperties(dom.value.panel, {
            style: isDetached.value ? {} : _getPanelPlacementStyle.getPanelPlacementStyle({
                panelPlacement: props.value.renderer.panelPlacement,
                container: dom.value.root,
                form: dom.value.form,
                environment: props.value.core.environment
            })
        });
    }
    function scheduleRender(state) {
        lastStateRef.current = state;
        var renderProps = {
            autocomplete: autocomplete1.value,
            autocompleteScopeApi: autocompleteScopeApi,
            classNames: props.value.renderer.classNames,
            components: props.value.renderer.components,
            container: props.value.renderer.container,
            html: html.value,
            dom: dom.value,
            panelContainer: isDetached.value ? dom.value.detachedContainer : props.value.renderer.panelContainer,
            propGetters: propGetters,
            state: lastStateRef.current,
            renderer: props.value.renderer.renderer
        };
        var render = !_autocompleteShared.getItemsCount(state) && !hasNoResultsSourceTemplateRef.current && props.value.renderer.renderNoResults || props.value.renderer.render;
        _render.renderSearchBox(renderProps);
        _render.renderPanel(render, renderProps);
    }
    runEffect(function() {
        var environmentProps = autocomplete1.value.getEnvironmentProps({
            formElement: dom.value.form,
            panelElement: dom.value.panel,
            inputElement: dom.value.input
        });
        _utils.setProperties(props.value.core.environment, environmentProps);
        return function() {
            _utils.setProperties(props.value.core.environment, Object.keys(environmentProps).reduce(function(acc, key) {
                return _objectSpread(_objectSpread({}, acc), {}, _defineProperty({}, key, undefined));
            }, {}));
        };
    });
    runEffect(function() {
        var panelContainerElement = isDetached.value ? props.value.core.environment.document.body : props.value.renderer.panelContainer;
        var panelElement = isDetached.value ? dom.value.detachedOverlay : dom.value.panel;
        if (isDetached.value && lastStateRef.current.isOpen) setIsModalOpen(true);
        scheduleRender(lastStateRef.current);
        return function() {
            if (panelContainerElement.contains(panelElement)) panelContainerElement.removeChild(panelElement);
        };
    });
    runEffect(function() {
        var containerElement = props.value.renderer.container;
        containerElement.appendChild(dom.value.root);
        return function() {
            containerElement.removeChild(dom.value.root);
        };
    });
    runEffect(function() {
        var debouncedRender = _autocompleteShared.debounce(function(_ref2) {
            var state = _ref2.state;
            scheduleRender(state);
        }, 0);
        onStateChangeRef.current = function(_ref3) {
            var state = _ref3.state, prevState = _ref3.prevState;
            if (isDetached.value && prevState.isOpen !== state.isOpen) setIsModalOpen(state.isOpen);
             // The outer DOM might have changed since the last time the panel was
            // positioned. The layout might have shifted vertically for instance.
            // It's therefore safer to re-calculate the panel position before opening
            // it again.
            if (!isDetached.value && state.isOpen && !prevState.isOpen) setPanelPosition();
             // We scroll to the top of the panel whenever the query changes (i.e. new
            // results come in) so that users don't have to.
            if (state.query !== prevState.query) {
                var scrollablePanels = props.value.core.environment.document.querySelectorAll('.aa-Panel--scrollable');
                scrollablePanels.forEach(function(scrollablePanel) {
                    if (scrollablePanel.scrollTop !== 0) scrollablePanel.scrollTop = 0;
                });
            }
            debouncedRender({
                state: state
            });
        };
        return function() {
            onStateChangeRef.current = undefined;
        };
    });
    runEffect(function() {
        var onResize = _autocompleteShared.debounce(function() {
            var previousIsDetached = isDetached.value;
            isDetached.value = props.value.core.environment.matchMedia(props.value.renderer.detachedMediaQuery).matches;
            if (previousIsDetached !== isDetached.value) update({});
            else requestAnimationFrame(setPanelPosition);
        }, 20);
        props.value.core.environment.addEventListener('resize', onResize);
        return function() {
            props.value.core.environment.removeEventListener('resize', onResize);
        };
    });
    runEffect(function() {
        if (!isDetached.value) return function() {};
        function toggleModalClassname(isActive) {
            dom.value.detachedContainer.classList.toggle('aa-DetachedContainer--modal', isActive);
        }
        function onChange(event) {
            toggleModalClassname(event.matches);
        }
        var isModalDetachedMql = props.value.core.environment.matchMedia(getComputedStyle(props.value.core.environment.document.documentElement).getPropertyValue('--aa-detached-modal-media-query'));
        toggleModalClassname(isModalDetachedMql.matches); // Prior to Safari 14, `MediaQueryList` isn't based on `EventTarget`,
        // so we must use `addListener` and `removeListener` to observe media query lists.
        // See https://developer.mozilla.org/en-US/docs/Web/API/MediaQueryList/addListener
        var hasModernEventListener = Boolean(isModalDetachedMql.addEventListener);
        hasModernEventListener ? isModalDetachedMql.addEventListener('change', onChange) : isModalDetachedMql.addListener(onChange);
        return function() {
            hasModernEventListener ? isModalDetachedMql.removeEventListener('change', onChange) : isModalDetachedMql.removeListener(onChange);
        };
    });
    runEffect(function() {
        requestAnimationFrame(setPanelPosition);
        return function() {};
    });
    function destroy() {
        cleanupEffects();
    }
    function update() {
        var updatedOptions = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : {};
        cleanupEffects();
        var _props$value$renderer = props.value.renderer, components = _props$value$renderer.components, rendererProps = _objectWithoutProperties(_props$value$renderer, _excluded);
        optionsRef.current = _utils.mergeDeep(rendererProps, props.value.core, {
            // We need to filter out default components so they can be replaced with
            // a new `renderer`, without getting rid of user components.
            // @MAJOR Deal with registering components with the same name as the
            // default ones. If we disallow overriding default components, we'd just
            // need to pass all `components` here.
            components: _utils.pickBy(components, function(_ref4) {
                var value = _ref4.value;
                return !value.hasOwnProperty('__autocomplete_componentName');
            }),
            initialState: lastStateRef.current
        }, updatedOptions);
        runReactives();
        runEffects();
        autocomplete1.value.refresh().then(function() {
            scheduleRender(lastStateRef.current);
        });
    }
    function setIsModalOpen(value) {
        requestAnimationFrame(function() {
            var prevValue = props.value.core.environment.document.body.contains(dom.value.detachedOverlay);
            if (value === prevValue) return;
            if (value) {
                props.value.core.environment.document.body.appendChild(dom.value.detachedOverlay);
                props.value.core.environment.document.body.classList.add('aa-Detached');
                dom.value.input.focus();
            } else {
                props.value.core.environment.document.body.removeChild(dom.value.detachedOverlay);
                props.value.core.environment.document.body.classList.remove('aa-Detached');
                autocomplete1.value.setQuery('');
                autocomplete1.value.refresh();
            }
        });
    }
    return _objectSpread(_objectSpread({}, autocompleteScopeApi), {}, {
        update: update,
        destroy: destroy
    });
}

},{"@algolia/autocomplete-core":"eH7jJ","@algolia/autocomplete-shared":"59T59","htm":"58er4","./createAutocompleteDom":"fEfqR","./createEffectWrapper":"71EHb","./createReactiveWrapper":"grTGt","./getDefaultOptions":"ei4ti","./getPanelPlacementStyle":"aVhxY","./render":"45iVn","./userAgents":"hqEHF","./utils":"fKU1x","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"eH7jJ":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _createAutocomplete = require("./createAutocomplete");
parcelHelpers.exportAll(_createAutocomplete, exports);
var _getDefaultProps = require("./getDefaultProps");
parcelHelpers.exportAll(_getDefaultProps, exports);
var _types = require("./types");
parcelHelpers.exportAll(_types, exports);

},{"./createAutocomplete":"SohsF","./getDefaultProps":"66IWD","./types":"79Luq","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"SohsF":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "createAutocomplete", ()=>createAutocomplete
);
var _checkOptions = require("./checkOptions");
var _createStore = require("./createStore");
var _getAutocompleteSetters = require("./getAutocompleteSetters");
var _getDefaultProps = require("./getDefaultProps");
var _getPropGetters = require("./getPropGetters");
var _metadata = require("./metadata");
var _onInput = require("./onInput");
var _stateReducer = require("./stateReducer");
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        enumerableOnly && (symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        })), keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = null != arguments[i] ? arguments[i] : {};
        i % 2 ? ownKeys(Object(source), !0).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
function createAutocomplete(options) {
    _checkOptions.checkOptions(options);
    var subscribers = [];
    var props = _getDefaultProps.getDefaultProps(options, subscribers);
    var store = _createStore.createStore(_stateReducer.stateReducer, props, onStoreStateChange);
    var setters = _getAutocompleteSetters.getAutocompleteSetters({
        store: store
    });
    var propGetters = _getPropGetters.getPropGetters(_objectSpread({
        props: props,
        refresh: refresh,
        store: store
    }, setters));
    function onStoreStateChange(_ref) {
        var prevState = _ref.prevState, state = _ref.state;
        props.onStateChange(_objectSpread({
            prevState: prevState,
            state: state,
            refresh: refresh
        }, setters));
    }
    function refresh() {
        return _onInput.onInput(_objectSpread({
            event: new Event('input'),
            nextState: {
                isOpen: store.getState().isOpen
            },
            props: props,
            query: store.getState().query,
            refresh: refresh,
            store: store
        }, setters));
    }
    props.plugins.forEach(function(plugin) {
        var _plugin$subscribe;
        return (_plugin$subscribe = plugin.subscribe) === null || _plugin$subscribe === void 0 ? void 0 : _plugin$subscribe.call(plugin, _objectSpread(_objectSpread({}, setters), {}, {
            refresh: refresh,
            onSelect: function onSelect(fn) {
                subscribers.push({
                    onSelect: fn
                });
            },
            onActive: function onActive(fn) {
                subscribers.push({
                    onActive: fn
                });
            }
        }));
    });
    _metadata.injectMetadata({
        metadata: _metadata.getMetadata({
            plugins: props.plugins,
            options: options
        }),
        environment: props.environment
    });
    return _objectSpread(_objectSpread({
        refresh: refresh
    }, propGetters), setters);
}

},{"./checkOptions":"dYRik","./createStore":"jHRzQ","./getAutocompleteSetters":"1FuiC","./getDefaultProps":"66IWD","./getPropGetters":"ghiHp","./metadata":"66SZT","./onInput":"6DlJz","./stateReducer":"iw5Pd","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"dYRik":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "checkOptions", ()=>checkOptions
);
var _autocompleteShared = require("@algolia/autocomplete-shared");
function checkOptions(options) {
    _autocompleteShared.warn(!options.debug, 'The `debug` option is meant for development debugging and should not be used in production.');
}

},{"@algolia/autocomplete-shared":"59T59","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"59T59":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _createRef = require("./createRef");
parcelHelpers.exportAll(_createRef, exports);
var _debounce = require("./debounce");
parcelHelpers.exportAll(_debounce, exports);
var _decycle = require("./decycle");
parcelHelpers.exportAll(_decycle, exports);
var _flatten = require("./flatten");
parcelHelpers.exportAll(_flatten, exports);
var _generateAutocompleteId = require("./generateAutocompleteId");
parcelHelpers.exportAll(_generateAutocompleteId, exports);
var _getAttributeValueByPath = require("./getAttributeValueByPath");
parcelHelpers.exportAll(_getAttributeValueByPath, exports);
var _getItemsCount = require("./getItemsCount");
parcelHelpers.exportAll(_getItemsCount, exports);
var _invariant = require("./invariant");
parcelHelpers.exportAll(_invariant, exports);
var _isEqual = require("./isEqual");
parcelHelpers.exportAll(_isEqual, exports);
var _maybePromise = require("./MaybePromise");
parcelHelpers.exportAll(_maybePromise, exports);
var _noop = require("./noop");
parcelHelpers.exportAll(_noop, exports);
var _userAgent = require("./UserAgent");
parcelHelpers.exportAll(_userAgent, exports);
var _userAgents = require("./userAgents");
parcelHelpers.exportAll(_userAgents, exports);
var _version = require("./version");
parcelHelpers.exportAll(_version, exports);
var _warn = require("./warn");
parcelHelpers.exportAll(_warn, exports);

},{"./createRef":"03oym","./debounce":"bf45i","./decycle":"9Ql5H","./flatten":"jZiCW","./generateAutocompleteId":"4baw4","./getAttributeValueByPath":"1o3Vw","./getItemsCount":"etmH8","./invariant":"cWfda","./isEqual":"74yQd","./MaybePromise":"4IWsu","./noop":"7T7MV","./UserAgent":"7KM8u","./userAgents":"4FPMn","./version":"j5UdY","./warn":"bcIkl","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"03oym":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "createRef", ()=>createRef
);
function createRef(initialValue) {
    return {
        current: initialValue
    };
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"bf45i":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "debounce", ()=>debounce
);
function debounce(fn, time) {
    var timerId = undefined;
    return function() {
        for(var _len = arguments.length, args = new Array(_len), _key = 0; _key < _len; _key++)args[_key] = arguments[_key];
        if (timerId) clearTimeout(timerId);
        timerId = setTimeout(function() {
            return fn.apply(void 0, args);
        }, time);
    };
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"9Ql5H":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/**
 * Decycles objects with circular references.
 * This is used to print cyclic structures in development environment only.
 */ parcelHelpers.export(exports, "decycle", ()=>decycle
);
function _slicedToArray(arr, i) {
    return _arrayWithHoles(arr) || _iterableToArrayLimit(arr, i) || _unsupportedIterableToArray(arr, i) || _nonIterableRest();
}
function _nonIterableRest() {
    throw new TypeError("Invalid attempt to destructure non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
}
function _unsupportedIterableToArray(o, minLen) {
    if (!o) return;
    if (typeof o === "string") return _arrayLikeToArray(o, minLen);
    var n = Object.prototype.toString.call(o).slice(8, -1);
    if (n === "Object" && o.constructor) n = o.constructor.name;
    if (n === "Map" || n === "Set") return Array.from(o);
    if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen);
}
function _arrayLikeToArray(arr, len) {
    if (len == null || len > arr.length) len = arr.length;
    for(var i = 0, arr2 = new Array(len); i < len; i++)arr2[i] = arr[i];
    return arr2;
}
function _iterableToArrayLimit(arr, i) {
    var _i = arr == null ? null : typeof Symbol !== "undefined" && arr[Symbol.iterator] || arr["@@iterator"];
    if (_i == null) return;
    var _arr = [];
    var _n = true;
    var _d = false;
    var _s, _e;
    try {
        for(_i = _i.call(arr); !(_n = (_s = _i.next()).done); _n = true){
            _arr.push(_s.value);
            if (i && _arr.length === i) break;
        }
    } catch (err) {
        _d = true;
        _e = err;
    } finally{
        try {
            if (!_n && _i["return"] != null) _i["return"]();
        } finally{
            if (_d) throw _e;
        }
    }
    return _arr;
}
function _arrayWithHoles(arr) {
    if (Array.isArray(arr)) return arr;
}
function _typeof(obj1) {
    return _typeof = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function(obj) {
        return typeof obj;
    } : function(obj) {
        return obj && "function" == typeof Symbol && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj;
    }, _typeof(obj1);
}
function decycle(obj) {
    var seen = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : new Set();
    if (!obj || _typeof(obj) !== 'object') return obj;
    if (seen.has(obj)) return '[Circular]';
    var newSeen = seen.add(obj);
    if (Array.isArray(obj)) return obj.map(function(x) {
        return decycle(x, newSeen);
    });
    return Object.fromEntries(Object.entries(obj).map(function(_ref) {
        var _ref2 = _slicedToArray(_ref, 2), key = _ref2[0], value = _ref2[1];
        return [
            key,
            decycle(value, newSeen)
        ];
    }));
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"jZiCW":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "flatten", ()=>flatten
);
function flatten(values) {
    return values.reduce(function(a, b) {
        return a.concat(b);
    }, []);
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"4baw4":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "generateAutocompleteId", ()=>generateAutocompleteId
);
var autocompleteId = 0;
function generateAutocompleteId() {
    return "autocomplete-".concat(autocompleteId++);
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"1o3Vw":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "getAttributeValueByPath", ()=>getAttributeValueByPath
);
function getAttributeValueByPath(record, path) {
    return path.reduce(function(current, key) {
        return current && current[key];
    }, record);
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"etmH8":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "getItemsCount", ()=>getItemsCount
);
function getItemsCount(state) {
    if (state.collections.length === 0) return 0;
    return state.collections.reduce(function(sum, collection) {
        return sum + collection.items.length;
    }, 0);
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"cWfda":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/**
 * Throws an error if the condition is not met in development mode.
 * This is used to make development a better experience to provide guidance as
 * to where the error comes from.
 */ parcelHelpers.export(exports, "invariant", ()=>invariant
);
function invariant(condition, message) {
    if (!condition) throw new Error("[Autocomplete] ".concat(typeof message === 'function' ? message() : message));
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"74yQd":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "isEqual", ()=>isEqual
);
function isPrimitive(obj) {
    return obj !== Object(obj);
}
function isEqual(first, second) {
    if (first === second) return true;
    if (isPrimitive(first) || isPrimitive(second) || typeof first === 'function' || typeof second === 'function') return first === second;
    if (Object.keys(first).length !== Object.keys(second).length) return false;
    for(var _i = 0, _Object$keys = Object.keys(first); _i < _Object$keys.length; _i++){
        var key = _Object$keys[_i];
        if (!(key in second)) return false;
        if (!isEqual(first[key], second[key])) return false;
    }
    return true;
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"4IWsu":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"7T7MV":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "noop", ()=>noop
);
var noop = function noop() {};

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"7KM8u":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"4FPMn":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "userAgents", ()=>userAgents
);
var _version = require("./version");
var userAgents = [
    {
        segment: 'autocomplete-core',
        version: _version.version
    }
];

},{"./version":"j5UdY","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"j5UdY":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "version", ()=>version
);
var version = '1.6.3';

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"bcIkl":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "warnCache", ()=>warnCache
);
/**
 * Logs a warning if the condition is not met.
 * This is used to log issues in development environment only.
 */ parcelHelpers.export(exports, "warn", ()=>warn
);
var warnCache = {
    current: {}
};
function warn(condition, message) {
    if (condition) return;
    var sanitizedMessage = message.trim();
    var hasAlreadyPrinted = warnCache.current[sanitizedMessage];
    if (!hasAlreadyPrinted) {
        warnCache.current[sanitizedMessage] = true; // eslint-disable-next-line no-console
        console.warn("[Autocomplete] ".concat(sanitizedMessage));
    }
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"jHRzQ":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "createStore", ()=>createStore
);
var _utils = require("./utils");
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        enumerableOnly && (symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        })), keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = null != arguments[i] ? arguments[i] : {};
        i % 2 ? ownKeys(Object(source), !0).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
function createStore(reducer, props, onStoreStateChange) {
    var state = props.initialState;
    return {
        getState: function getState() {
            return state;
        },
        dispatch: function dispatch(action, payload) {
            var prevState = _objectSpread({}, state);
            state = reducer(state, {
                type: action,
                props: props,
                payload: payload
            });
            onStoreStateChange({
                state: state,
                prevState: prevState
            });
        },
        pendingRequests: _utils.createCancelablePromiseList()
    };
}

},{"./utils":"gd60Y","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"gd60Y":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _createCancelablePromise = require("./createCancelablePromise");
parcelHelpers.exportAll(_createCancelablePromise, exports);
var _createCancelablePromiseList = require("./createCancelablePromiseList");
parcelHelpers.exportAll(_createCancelablePromiseList, exports);
var _createConcurrentSafePromise = require("./createConcurrentSafePromise");
parcelHelpers.exportAll(_createConcurrentSafePromise, exports);
var _getNextActiveItemId = require("./getNextActiveItemId");
parcelHelpers.exportAll(_getNextActiveItemId, exports);
var _getNormalizedSources = require("./getNormalizedSources");
parcelHelpers.exportAll(_getNormalizedSources, exports);
var _getActiveItem = require("./getActiveItem");
parcelHelpers.exportAll(_getActiveItem, exports);
var _isOrContainsNode = require("./isOrContainsNode");
parcelHelpers.exportAll(_isOrContainsNode, exports);
var _isSamsung = require("./isSamsung");
parcelHelpers.exportAll(_isSamsung, exports);
var _mapToAlgoliaResponse = require("./mapToAlgoliaResponse");
parcelHelpers.exportAll(_mapToAlgoliaResponse, exports);

},{"./createCancelablePromise":"gLzIw","./createCancelablePromiseList":"8etN5","./createConcurrentSafePromise":"4Cc3M","./getNextActiveItemId":"g57RT","./getNormalizedSources":"8mENM","./getActiveItem":"bFTMy","./isOrContainsNode":"5AASb","./isSamsung":"8H5bw","./mapToAlgoliaResponse":"gS8c5","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"gLzIw":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "createCancelablePromise", ()=>createCancelablePromise
);
parcelHelpers.export(exports, "cancelable", ()=>cancelable
);
function createInternalCancelablePromise(promise, initialState) {
    var state = initialState;
    return {
        then: function then(onfulfilled, onrejected) {
            return createInternalCancelablePromise(promise.then(createCallback(onfulfilled, state, promise), createCallback(onrejected, state, promise)), state);
        },
        catch: function _catch(onrejected) {
            return createInternalCancelablePromise(promise.catch(createCallback(onrejected, state, promise)), state);
        },
        finally: function _finally(onfinally) {
            if (onfinally) state.onCancelList.push(onfinally);
            return createInternalCancelablePromise(promise.finally(createCallback(onfinally && function() {
                state.onCancelList = [];
                return onfinally();
            }, state, promise)), state);
        },
        cancel: function cancel() {
            state.isCanceled = true;
            var callbacks = state.onCancelList;
            state.onCancelList = [];
            callbacks.forEach(function(callback) {
                callback();
            });
        },
        isCanceled: function isCanceled() {
            return state.isCanceled === true;
        }
    };
}
function createCancelablePromise(executor) {
    return createInternalCancelablePromise(new Promise(function(resolve, reject) {
        return executor(resolve, reject);
    }), {
        isCanceled: false,
        onCancelList: []
    });
}
createCancelablePromise.resolve = function(value) {
    return cancelable(Promise.resolve(value));
};
createCancelablePromise.reject = function(reason) {
    return cancelable(Promise.reject(reason));
};
function cancelable(promise) {
    return createInternalCancelablePromise(promise, {
        isCanceled: false,
        onCancelList: []
    });
}
function createCallback(onResult, state, fallback) {
    if (!onResult) return fallback;
    return function callback(arg) {
        if (state.isCanceled) return arg;
        return onResult(arg);
    };
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"8etN5":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "createCancelablePromiseList", ()=>createCancelablePromiseList
);
function createCancelablePromiseList() {
    var list = [];
    return {
        add: function add(cancelablePromise) {
            list.push(cancelablePromise);
            return cancelablePromise.finally(function() {
                list = list.filter(function(item) {
                    return item !== cancelablePromise;
                });
            });
        },
        cancelAll: function cancelAll() {
            list.forEach(function(promise) {
                return promise.cancel();
            });
        },
        isEmpty: function isEmpty() {
            return list.length === 0;
        }
    };
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"4Cc3M":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/**
 * Creates a runner that executes promises in a concurrent-safe way.
 *
 * This is useful to prevent older promises to resolve after a newer promise,
 * otherwise resulting in stale resolved values.
 */ parcelHelpers.export(exports, "createConcurrentSafePromise", ()=>createConcurrentSafePromise
);
function createConcurrentSafePromise() {
    var basePromiseId = -1;
    var latestResolvedId = -1;
    var latestResolvedValue = undefined;
    return function runConcurrentSafePromise(promise) {
        basePromiseId++;
        var currentPromiseId = basePromiseId;
        return Promise.resolve(promise).then(function(x) {
            // The promise might take too long to resolve and get outdated. This would
            // result in resolving stale values.
            // When this happens, we ignore the promise value and return the one
            // coming from the latest resolved value.
            //
            // +----------------------------------+
            // |        100ms                     |
            // | run(1) +--->  R1                 |
            // |        300ms                     |
            // | run(2) +-------------> R2 (SKIP) |
            // |        200ms                     |
            // | run(3) +--------> R3             |
            // +----------------------------------+
            if (latestResolvedValue && currentPromiseId < latestResolvedId) return latestResolvedValue;
            latestResolvedId = currentPromiseId;
            latestResolvedValue = x;
            return x;
        });
    };
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"g57RT":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/**
 * Returns the next active item ID from the current state.
 *
 * We allow circular keyboard navigation from the base index.
 * The base index can either be `null` (nothing is highlighted) or `0`
 * (the first item is highlighted).
 * The base index is allowed to get assigned `null` only if
 * `props.defaultActiveItemId` is `null`. This pattern allows to "stop"
 * by the actual query before navigating to other suggestions as seen on
 * Google or Amazon.
 *
 * @param moveAmount The offset to increment (or decrement) the last index
 * @param baseIndex The current index to compute the next index from
 * @param itemCount The number of items
 * @param defaultActiveItemId The default active index to fallback to
 */ parcelHelpers.export(exports, "getNextActiveItemId", ()=>getNextActiveItemId
);
function getNextActiveItemId(moveAmount, baseIndex, itemCount, defaultActiveItemId) {
    if (!itemCount) return null;
    if (moveAmount < 0 && (baseIndex === null || defaultActiveItemId !== null && baseIndex === 0)) return itemCount + moveAmount;
    var numericIndex = (baseIndex === null ? -1 : baseIndex) + moveAmount;
    if (numericIndex <= -1 || numericIndex >= itemCount) return defaultActiveItemId === null ? null : 0;
    return numericIndex;
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"8mENM":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "getNormalizedSources", ()=>getNormalizedSources
);
var _autocompleteShared = require("@algolia/autocomplete-shared");
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        enumerableOnly && (symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        })), keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = null != arguments[i] ? arguments[i] : {};
        i % 2 ? ownKeys(Object(source), !0).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
function _typeof(obj1) {
    return _typeof = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function(obj) {
        return typeof obj;
    } : function(obj) {
        return obj && "function" == typeof Symbol && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj;
    }, _typeof(obj1);
}
function getNormalizedSources(getSources, params) {
    var seenSourceIds = [];
    return Promise.resolve(getSources(params)).then(function(sources) {
        _autocompleteShared.invariant(Array.isArray(sources), function() {
            return "The `getSources` function must return an array of sources but returned type ".concat(JSON.stringify(_typeof(sources)), ":\n\n").concat(JSON.stringify(_autocompleteShared.decycle(sources), null, 2));
        });
        return Promise.all(sources // We allow `undefined` and `false` sources to allow users to use
        // `Boolean(query) && source` (=> `false`).
        // We need to remove these values at this point.
        .filter(function(maybeSource) {
            return Boolean(maybeSource);
        }).map(function(source) {
            _autocompleteShared.invariant(typeof source.sourceId === 'string', 'A source must provide a `sourceId` string.');
            if (seenSourceIds.includes(source.sourceId)) throw new Error("[Autocomplete] The `sourceId` ".concat(JSON.stringify(source.sourceId), " is not unique."));
            seenSourceIds.push(source.sourceId);
            var normalizedSource = _objectSpread({
                getItemInputValue: function getItemInputValue(_ref) {
                    var state = _ref.state;
                    return state.query;
                },
                getItemUrl: function getItemUrl() {
                    return undefined;
                },
                onSelect: function onSelect(_ref2) {
                    var setIsOpen = _ref2.setIsOpen;
                    setIsOpen(false);
                },
                onActive: _autocompleteShared.noop
            }, source);
            return Promise.resolve(normalizedSource);
        }));
    });
}

},{"@algolia/autocomplete-shared":"59T59","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"bFTMy":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "getActiveItem", ()=>getActiveItem
);
// We don't have access to the autocomplete source when we call `onKeyDown`
// or `onClick` because those are native browser events.
// However, we can get the source from the suggestion index.
function getCollectionFromActiveItemId(state) {
    // Given 3 sources with respectively 1, 2 and 3 suggestions: [1, 2, 3]
    // We want to get the accumulated counts:
    // [1, 1 + 2, 1 + 2 + 3] = [1, 3, 3 + 3] = [1, 3, 6]
    var accumulatedCollectionsCount = state.collections.map(function(collections) {
        return collections.items.length;
    }).reduce(function(acc, collectionsCount, index) {
        var previousValue = acc[index - 1] || 0;
        var nextValue = previousValue + collectionsCount;
        acc.push(nextValue);
        return acc;
    }, []); // Based on the accumulated counts, we can infer the index of the suggestion.
    var collectionIndex = accumulatedCollectionsCount.reduce(function(acc, current) {
        if (current <= state.activeItemId) return acc + 1;
        return acc;
    }, 0);
    return state.collections[collectionIndex];
}
/**
 * Gets the highlighted index relative to a suggestion object (not the absolute
 * highlighted index).
 *
 * Example:
 *  [['a', 'b'], ['c', 'd', 'e'], ['f']]
 *                      ↑
 *         (absolute: 3, relative: 1)
 */ function getRelativeActiveItemId(_ref) {
    var state = _ref.state, collection = _ref.collection;
    var isOffsetFound = false;
    var counter = 0;
    var previousItemsOffset = 0;
    while(isOffsetFound === false){
        var currentCollection = state.collections[counter];
        if (currentCollection === collection) {
            isOffsetFound = true;
            break;
        }
        previousItemsOffset += currentCollection.items.length;
        counter++;
    }
    return state.activeItemId - previousItemsOffset;
}
function getActiveItem(state) {
    var collection = getCollectionFromActiveItemId(state);
    if (!collection) return null;
    var item = collection.items[getRelativeActiveItemId({
        state: state,
        collection: collection
    })];
    var source = collection.source;
    var itemInputValue = source.getItemInputValue({
        item: item,
        state: state
    });
    var itemUrl = source.getItemUrl({
        item: item,
        state: state
    });
    return {
        item: item,
        itemInputValue: itemInputValue,
        itemUrl: itemUrl,
        source: source
    };
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"5AASb":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "isOrContainsNode", ()=>isOrContainsNode
);
function isOrContainsNode(parent, child) {
    return parent === child || parent.contains(child);
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"8H5bw":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "isSamsung", ()=>isSamsung
);
var regex = /((gt|sm)-|galaxy nexus)|samsung[- ]/i;
function isSamsung(userAgent) {
    return Boolean(userAgent && userAgent.match(regex));
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"gS8c5":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "mapToAlgoliaResponse", ()=>mapToAlgoliaResponse
);
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        enumerableOnly && (symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        })), keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = null != arguments[i] ? arguments[i] : {};
        i % 2 ? ownKeys(Object(source), !0).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
function mapToAlgoliaResponse(rawResults) {
    var results = rawResults.map(function(result) {
        var _hits;
        return _objectSpread(_objectSpread({}, result), {}, {
            hits: (_hits = result.hits) === null || _hits === void 0 ? void 0 : _hits.map(function(hit) {
                // Bring support for the Insights plugin.
                return _objectSpread(_objectSpread({}, hit), {}, {
                    __autocomplete_indexName: result.index,
                    __autocomplete_queryID: result.queryID
                });
            })
        });
    });
    return {
        results: results,
        hits: results.map(function(result) {
            return result.hits;
        }).filter(Boolean),
        facetHits: results.map(function(result) {
            var _facetHits;
            return (_facetHits = result.facetHits) === null || _facetHits === void 0 ? void 0 : _facetHits.map(function(facetHit) {
                // Bring support for the highlighting components.
                return {
                    label: facetHit.value,
                    count: facetHit.count,
                    _highlightResult: {
                        label: {
                            value: facetHit.highlighted
                        }
                    }
                };
            });
        }).filter(Boolean)
    };
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"1FuiC":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "getAutocompleteSetters", ()=>getAutocompleteSetters
);
var _autocompleteShared = require("@algolia/autocomplete-shared");
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        enumerableOnly && (symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        })), keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = null != arguments[i] ? arguments[i] : {};
        i % 2 ? ownKeys(Object(source), !0).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
function getAutocompleteSetters(_ref) {
    var store = _ref.store;
    var setActiveItemId = function setActiveItemId(value) {
        store.dispatch('setActiveItemId', value);
    };
    var setQuery = function setQuery(value) {
        store.dispatch('setQuery', value);
    };
    var setCollections = function setCollections(rawValue) {
        var baseItemId = 0;
        var value = rawValue.map(function(collection) {
            return _objectSpread(_objectSpread({}, collection), {}, {
                // We flatten the stored items to support calling `getAlgoliaResults`
                // from the source itself.
                items: _autocompleteShared.flatten(collection.items).map(function(item) {
                    return _objectSpread(_objectSpread({}, item), {}, {
                        __autocomplete_id: baseItemId++
                    });
                })
            });
        });
        store.dispatch('setCollections', value);
    };
    var setIsOpen = function setIsOpen(value) {
        store.dispatch('setIsOpen', value);
    };
    var setStatus = function setStatus(value) {
        store.dispatch('setStatus', value);
    };
    var setContext = function setContext(value) {
        store.dispatch('setContext', value);
    };
    return {
        setActiveItemId: setActiveItemId,
        setQuery: setQuery,
        setCollections: setCollections,
        setIsOpen: setIsOpen,
        setStatus: setStatus,
        setContext: setContext
    };
}

},{"@algolia/autocomplete-shared":"59T59","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"66IWD":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "getDefaultProps", ()=>getDefaultProps
);
var _autocompleteShared = require("@algolia/autocomplete-shared");
var _utils = require("./utils");
function _toConsumableArray(arr) {
    return _arrayWithoutHoles(arr) || _iterableToArray(arr) || _unsupportedIterableToArray(arr) || _nonIterableSpread();
}
function _nonIterableSpread() {
    throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
}
function _unsupportedIterableToArray(o, minLen) {
    if (!o) return;
    if (typeof o === "string") return _arrayLikeToArray(o, minLen);
    var n = Object.prototype.toString.call(o).slice(8, -1);
    if (n === "Object" && o.constructor) n = o.constructor.name;
    if (n === "Map" || n === "Set") return Array.from(o);
    if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen);
}
function _iterableToArray(iter) {
    if (typeof Symbol !== "undefined" && iter[Symbol.iterator] != null || iter["@@iterator"] != null) return Array.from(iter);
}
function _arrayWithoutHoles(arr) {
    if (Array.isArray(arr)) return _arrayLikeToArray(arr);
}
function _arrayLikeToArray(arr, len) {
    if (len == null || len > arr.length) len = arr.length;
    for(var i = 0, arr2 = new Array(len); i < len; i++)arr2[i] = arr[i];
    return arr2;
}
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        enumerableOnly && (symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        })), keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = null != arguments[i] ? arguments[i] : {};
        i % 2 ? ownKeys(Object(source), !0).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
function getDefaultProps(props, pluginSubscribers) {
    var _props$id;
    /* eslint-disable no-restricted-globals */ var environment = typeof window !== 'undefined' ? window : {};
    /* eslint-enable no-restricted-globals */ var plugins = props.plugins || [];
    return _objectSpread(_objectSpread({
        debug: false,
        openOnFocus: false,
        placeholder: '',
        autoFocus: false,
        defaultActiveItemId: null,
        stallThreshold: 300,
        environment: environment,
        shouldPanelOpen: function shouldPanelOpen(_ref) {
            var state = _ref.state;
            return _autocompleteShared.getItemsCount(state) > 0;
        },
        reshape: function reshape(_ref2) {
            var sources = _ref2.sources;
            return sources;
        }
    }, props), {}, {
        // Since `generateAutocompleteId` triggers a side effect (it increments
        // an internal counter), we don't want to execute it if unnecessary.
        id: (_props$id = props.id) !== null && _props$id !== void 0 ? _props$id : _autocompleteShared.generateAutocompleteId(),
        plugins: plugins,
        // The following props need to be deeply defaulted.
        initialState: _objectSpread({
            activeItemId: null,
            query: '',
            completion: null,
            collections: [],
            isOpen: false,
            status: 'idle',
            context: {}
        }, props.initialState),
        onStateChange: function onStateChange(params) {
            var _props$onStateChange;
            (_props$onStateChange = props.onStateChange) === null || _props$onStateChange === void 0 || _props$onStateChange.call(props, params);
            plugins.forEach(function(x) {
                var _x$onStateChange;
                return (_x$onStateChange = x.onStateChange) === null || _x$onStateChange === void 0 ? void 0 : _x$onStateChange.call(x, params);
            });
        },
        onSubmit: function onSubmit(params) {
            var _props$onSubmit;
            (_props$onSubmit = props.onSubmit) === null || _props$onSubmit === void 0 || _props$onSubmit.call(props, params);
            plugins.forEach(function(x) {
                var _x$onSubmit;
                return (_x$onSubmit = x.onSubmit) === null || _x$onSubmit === void 0 ? void 0 : _x$onSubmit.call(x, params);
            });
        },
        onReset: function onReset(params) {
            var _props$onReset;
            (_props$onReset = props.onReset) === null || _props$onReset === void 0 || _props$onReset.call(props, params);
            plugins.forEach(function(x) {
                var _x$onReset;
                return (_x$onReset = x.onReset) === null || _x$onReset === void 0 ? void 0 : _x$onReset.call(x, params);
            });
        },
        getSources: function getSources1(params1) {
            return Promise.all([].concat(_toConsumableArray(plugins.map(function(plugin) {
                return plugin.getSources;
            })), [
                props.getSources
            ]).filter(Boolean).map(function(getSources) {
                return _utils.getNormalizedSources(getSources, params1);
            })).then(function(nested) {
                return _autocompleteShared.flatten(nested);
            }).then(function(sources) {
                return sources.map(function(source) {
                    return _objectSpread(_objectSpread({}, source), {}, {
                        onSelect: function onSelect(params) {
                            source.onSelect(params);
                            pluginSubscribers.forEach(function(x) {
                                var _x$onSelect;
                                return (_x$onSelect = x.onSelect) === null || _x$onSelect === void 0 ? void 0 : _x$onSelect.call(x, params);
                            });
                        },
                        onActive: function onActive(params) {
                            source.onActive(params);
                            pluginSubscribers.forEach(function(x) {
                                var _x$onActive;
                                return (_x$onActive = x.onActive) === null || _x$onActive === void 0 ? void 0 : _x$onActive.call(x, params);
                            });
                        }
                    });
                });
            });
        },
        navigator: _objectSpread({
            navigate: function navigate(_ref3) {
                var itemUrl = _ref3.itemUrl;
                environment.location.assign(itemUrl);
            },
            navigateNewTab: function navigateNewTab(_ref4) {
                var itemUrl = _ref4.itemUrl;
                var windowReference = environment.open(itemUrl, '_blank', 'noopener');
                windowReference === null || windowReference === void 0 || windowReference.focus();
            },
            navigateNewWindow: function navigateNewWindow(_ref5) {
                var itemUrl = _ref5.itemUrl;
                environment.open(itemUrl, '_blank', 'noopener');
            }
        }, props.navigator)
    });
}

},{"@algolia/autocomplete-shared":"59T59","./utils":"gd60Y","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"ghiHp":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "getPropGetters", ()=>getPropGetters
);
var _onInput = require("./onInput");
var _onKeyDown = require("./onKeyDown");
var _utils = require("./utils");
var _excluded = [
    "props",
    "refresh",
    "store"
], _excluded2 = [
    "inputElement",
    "formElement",
    "panelElement"
], _excluded3 = [
    "inputElement"
], _excluded4 = [
    "inputElement",
    "maxLength"
], _excluded5 = [
    "item",
    "source"
];
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        enumerableOnly && (symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        })), keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = null != arguments[i] ? arguments[i] : {};
        i % 2 ? ownKeys(Object(source), !0).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
function _objectWithoutProperties(source, excluded) {
    if (source == null) return {};
    var target = _objectWithoutPropertiesLoose(source, excluded);
    var key, i;
    if (Object.getOwnPropertySymbols) {
        var sourceSymbolKeys = Object.getOwnPropertySymbols(source);
        for(i = 0; i < sourceSymbolKeys.length; i++){
            key = sourceSymbolKeys[i];
            if (excluded.indexOf(key) >= 0) continue;
            if (!Object.prototype.propertyIsEnumerable.call(source, key)) continue;
            target[key] = source[key];
        }
    }
    return target;
}
function _objectWithoutPropertiesLoose(source, excluded) {
    if (source == null) return {};
    var target = {};
    var sourceKeys = Object.keys(source);
    var key, i;
    for(i = 0; i < sourceKeys.length; i++){
        key = sourceKeys[i];
        if (excluded.indexOf(key) >= 0) continue;
        target[key] = source[key];
    }
    return target;
}
function getPropGetters(_ref) {
    var props = _ref.props, refresh = _ref.refresh, store = _ref.store, setters = _objectWithoutProperties(_ref, _excluded);
    var getEnvironmentProps = function getEnvironmentProps(providedProps) {
        var inputElement = providedProps.inputElement, formElement = providedProps.formElement, panelElement = providedProps.panelElement, rest = _objectWithoutProperties(providedProps, _excluded2);
        return _objectSpread({
            // On touch devices, we do not rely on the native `blur` event of the
            // input to close the panel, but rather on a custom `touchstart` event
            // outside of the autocomplete elements.
            // This ensures a working experience on mobile because we blur the input
            // on touch devices when the user starts scrolling (`touchmove`).
            // @TODO: support cases where there are multiple Autocomplete instances.
            // Right now, a second instance makes this computation return false.
            onTouchStart: function onTouchStart(event) {
                // The `onTouchStart` event shouldn't trigger the `blur` handler when
                // it's not an interaction with Autocomplete. We detect it with the
                // following heuristics:
                // - the panel is closed AND there are no pending requests
                //   (no interaction with the autocomplete, no future state updates)
                // - OR the touched target is the input element (should open the panel)
                var isAutocompleteInteraction = store.getState().isOpen || !store.pendingRequests.isEmpty();
                if (!isAutocompleteInteraction || event.target === inputElement) return;
                var isTargetWithinAutocomplete = [
                    formElement,
                    panelElement
                ].some(function(contextNode) {
                    return _utils.isOrContainsNode(contextNode, event.target);
                });
                if (isTargetWithinAutocomplete === false) {
                    store.dispatch('blur', null); // If requests are still pending when the user closes the panel, they
                    // could reopen the panel once they resolve.
                    // We want to prevent any subsequent query from reopening the panel
                    // because it would result in an unsolicited UI behavior.
                    if (!props.debug) store.pendingRequests.cancelAll();
                }
            },
            // When scrolling on touch devices (mobiles, tablets, etc.), we want to
            // mimic the native platform behavior where the input is blurred to
            // hide the virtual keyboard. This gives more vertical space to
            // discover all the suggestions showing up in the panel.
            onTouchMove: function onTouchMove(event) {
                if (store.getState().isOpen === false || inputElement !== props.environment.document.activeElement || event.target === inputElement) return;
                inputElement.blur();
            }
        }, rest);
    };
    var getRootProps = function getRootProps(rest) {
        return _objectSpread({
            role: 'combobox',
            'aria-expanded': store.getState().isOpen,
            'aria-haspopup': 'listbox',
            'aria-owns': store.getState().isOpen ? "".concat(props.id, "-list") : undefined,
            'aria-labelledby': "".concat(props.id, "-label")
        }, rest);
    };
    var getFormProps = function getFormProps(providedProps) {
        var inputElement = providedProps.inputElement, rest = _objectWithoutProperties(providedProps, _excluded3);
        return _objectSpread({
            action: '',
            noValidate: true,
            role: 'search',
            onSubmit: function onSubmit(event) {
                var _providedProps$inputE;
                event.preventDefault();
                props.onSubmit(_objectSpread({
                    event: event,
                    refresh: refresh,
                    state: store.getState()
                }, setters));
                store.dispatch('submit', null);
                (_providedProps$inputE = providedProps.inputElement) === null || _providedProps$inputE === void 0 || _providedProps$inputE.blur();
            },
            onReset: function onReset(event) {
                var _providedProps$inputE2;
                event.preventDefault();
                props.onReset(_objectSpread({
                    event: event,
                    refresh: refresh,
                    state: store.getState()
                }, setters));
                store.dispatch('reset', null);
                (_providedProps$inputE2 = providedProps.inputElement) === null || _providedProps$inputE2 === void 0 || _providedProps$inputE2.focus();
            }
        }, rest);
    };
    var getInputProps = function getInputProps(providedProps) {
        var _props$environment$na;
        function onFocus(event) {
            // We want to trigger a query when `openOnFocus` is true
            // because the panel should open with the current query.
            if (props.openOnFocus || Boolean(store.getState().query)) _onInput.onInput(_objectSpread({
                event: event,
                props: props,
                query: store.getState().completion || store.getState().query,
                refresh: refresh,
                store: store
            }, setters));
            store.dispatch('focus', null);
        }
        var isTouchDevice = 'ontouchstart' in props.environment;
        var _ref2 = providedProps || {}, inputElement = _ref2.inputElement, _ref2$maxLength = _ref2.maxLength, maxLength = _ref2$maxLength === void 0 ? 512 : _ref2$maxLength, rest = _objectWithoutProperties(_ref2, _excluded4);
        var activeItem = _utils.getActiveItem(store.getState());
        var userAgent = (_props$environment$na = props.environment.navigator) === null || _props$environment$na === void 0 ? void 0 : _props$environment$na.userAgent;
        var shouldFallbackKeyHint = _utils.isSamsung(userAgent);
        var enterKeyHint = activeItem !== null && activeItem !== void 0 && activeItem.itemUrl && !shouldFallbackKeyHint ? 'go' : 'search';
        return _objectSpread({
            'aria-autocomplete': 'both',
            'aria-activedescendant': store.getState().isOpen && store.getState().activeItemId !== null ? "".concat(props.id, "-item-").concat(store.getState().activeItemId) : undefined,
            'aria-controls': store.getState().isOpen ? "".concat(props.id, "-list") : undefined,
            'aria-labelledby': "".concat(props.id, "-label"),
            value: store.getState().completion || store.getState().query,
            id: "".concat(props.id, "-input"),
            autoComplete: 'off',
            autoCorrect: 'off',
            autoCapitalize: 'off',
            enterKeyHint: enterKeyHint,
            spellCheck: 'false',
            autoFocus: props.autoFocus,
            placeholder: props.placeholder,
            maxLength: maxLength,
            type: 'search',
            onChange: function onChange(event) {
                _onInput.onInput(_objectSpread({
                    event: event,
                    props: props,
                    query: event.currentTarget.value.slice(0, maxLength),
                    refresh: refresh,
                    store: store
                }, setters));
            },
            onKeyDown: function onKeyDown(event) {
                _onKeyDown.onKeyDown(_objectSpread({
                    event: event,
                    props: props,
                    refresh: refresh,
                    store: store
                }, setters));
            },
            onFocus: onFocus,
            onBlur: function onBlur() {
                // We do rely on the `blur` event on touch devices.
                // See explanation in `onTouchStart`.
                if (!isTouchDevice) {
                    store.dispatch('blur', null); // If requests are still pending when the user closes the panel, they
                    // could reopen the panel once they resolve.
                    // We want to prevent any subsequent query from reopening the panel
                    // because it would result in an unsolicited UI behavior.
                    if (!props.debug) store.pendingRequests.cancelAll();
                }
            },
            onClick: function onClick(event) {
                // When the panel is closed and you click on the input while
                // the input is focused, the `onFocus` event is not triggered
                // (default browser behavior).
                // In an autocomplete context, it makes sense to open the panel in this
                // case.
                // We mimic this event by catching the `onClick` event which
                // triggers the `onFocus` for the panel to open.
                if (providedProps.inputElement === props.environment.document.activeElement && !store.getState().isOpen) onFocus(event);
            }
        }, rest);
    };
    var getLabelProps = function getLabelProps(rest) {
        return _objectSpread({
            htmlFor: "".concat(props.id, "-input"),
            id: "".concat(props.id, "-label")
        }, rest);
    };
    var getListProps = function getListProps(rest) {
        return _objectSpread({
            role: 'listbox',
            'aria-labelledby': "".concat(props.id, "-label"),
            id: "".concat(props.id, "-list")
        }, rest);
    };
    var getPanelProps = function getPanelProps(rest) {
        return _objectSpread({
            onMouseDown: function onMouseDown(event) {
                // Prevents the `activeElement` from being changed to the panel so
                // that the blur event is not triggered, otherwise it closes the
                // panel.
                event.preventDefault();
            },
            onMouseLeave: function onMouseLeave() {
                store.dispatch('mouseleave', null);
            }
        }, rest);
    };
    var getItemProps = function getItemProps(providedProps) {
        var item = providedProps.item, source = providedProps.source, rest = _objectWithoutProperties(providedProps, _excluded5);
        return _objectSpread({
            id: "".concat(props.id, "-item-").concat(item.__autocomplete_id),
            role: 'option',
            'aria-selected': store.getState().activeItemId === item.__autocomplete_id,
            onMouseMove: function onMouseMove(event) {
                if (item.__autocomplete_id === store.getState().activeItemId) return;
                store.dispatch('mousemove', item.__autocomplete_id);
                var activeItem = _utils.getActiveItem(store.getState());
                if (store.getState().activeItemId !== null && activeItem) {
                    var _item = activeItem.item, itemInputValue = activeItem.itemInputValue, itemUrl = activeItem.itemUrl, _source = activeItem.source;
                    _source.onActive(_objectSpread({
                        event: event,
                        item: _item,
                        itemInputValue: itemInputValue,
                        itemUrl: itemUrl,
                        refresh: refresh,
                        source: _source,
                        state: store.getState()
                    }, setters));
                }
            },
            onMouseDown: function onMouseDown(event) {
                // Prevents the `activeElement` from being changed to the item so it
                // can remain with the current `activeElement`.
                event.preventDefault();
            },
            onClick: function onClick(event) {
                var itemInputValue = source.getItemInputValue({
                    item: item,
                    state: store.getState()
                });
                var itemUrl = source.getItemUrl({
                    item: item,
                    state: store.getState()
                }); // If `getItemUrl` is provided, it means that the suggestion
                // is a link, not plain text that aims at updating the query.
                // We can therefore skip the state change because it will update
                // the `activeItemId`, resulting in a UI flash, especially
                // noticeable on mobile.
                var runPreCommand = itemUrl ? Promise.resolve() : _onInput.onInput(_objectSpread({
                    event: event,
                    nextState: {
                        isOpen: false
                    },
                    props: props,
                    query: itemInputValue,
                    refresh: refresh,
                    store: store
                }, setters));
                runPreCommand.then(function() {
                    source.onSelect(_objectSpread({
                        event: event,
                        item: item,
                        itemInputValue: itemInputValue,
                        itemUrl: itemUrl,
                        refresh: refresh,
                        source: source,
                        state: store.getState()
                    }, setters));
                });
            }
        }, rest);
    };
    return {
        getEnvironmentProps: getEnvironmentProps,
        getRootProps: getRootProps,
        getFormProps: getFormProps,
        getLabelProps: getLabelProps,
        getInputProps: getInputProps,
        getPanelProps: getPanelProps,
        getListProps: getListProps,
        getItemProps: getItemProps
    };
}

},{"./onInput":"6DlJz","./onKeyDown":"ahwBt","./utils":"gd60Y","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"6DlJz":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "onInput", ()=>onInput
);
var _reshape = require("./reshape");
var _resolve = require("./resolve");
var _utils = require("./utils");
var _excluded = [
    "event",
    "nextState",
    "props",
    "query",
    "refresh",
    "store"
];
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        enumerableOnly && (symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        })), keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = null != arguments[i] ? arguments[i] : {};
        i % 2 ? ownKeys(Object(source), !0).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
function _objectWithoutProperties(source, excluded) {
    if (source == null) return {};
    var target = _objectWithoutPropertiesLoose(source, excluded);
    var key, i;
    if (Object.getOwnPropertySymbols) {
        var sourceSymbolKeys = Object.getOwnPropertySymbols(source);
        for(i = 0; i < sourceSymbolKeys.length; i++){
            key = sourceSymbolKeys[i];
            if (excluded.indexOf(key) >= 0) continue;
            if (!Object.prototype.propertyIsEnumerable.call(source, key)) continue;
            target[key] = source[key];
        }
    }
    return target;
}
function _objectWithoutPropertiesLoose(source, excluded) {
    if (source == null) return {};
    var target = {};
    var sourceKeys = Object.keys(source);
    var key, i;
    for(i = 0; i < sourceKeys.length; i++){
        key = sourceKeys[i];
        if (excluded.indexOf(key) >= 0) continue;
        target[key] = source[key];
    }
    return target;
}
var lastStalledId = null;
var runConcurrentSafePromise = _utils.createConcurrentSafePromise();
function onInput(_ref) {
    var event = _ref.event, _ref$nextState = _ref.nextState, nextState = _ref$nextState === void 0 ? {} : _ref$nextState, props = _ref.props, query = _ref.query, refresh = _ref.refresh, store = _ref.store, setters = _objectWithoutProperties(_ref, _excluded);
    if (lastStalledId) props.environment.clearTimeout(lastStalledId);
    var setCollections = setters.setCollections, setIsOpen = setters.setIsOpen, setQuery = setters.setQuery, setActiveItemId = setters.setActiveItemId, setStatus = setters.setStatus;
    setQuery(query);
    setActiveItemId(props.defaultActiveItemId);
    if (!query && props.openOnFocus === false) {
        var _nextState$isOpen;
        var collections = store.getState().collections.map(function(collection) {
            return _objectSpread(_objectSpread({}, collection), {}, {
                items: []
            });
        });
        setStatus('idle');
        setCollections(collections);
        setIsOpen((_nextState$isOpen = nextState.isOpen) !== null && _nextState$isOpen !== void 0 ? _nextState$isOpen : props.shouldPanelOpen({
            state: store.getState()
        })); // We make sure to update the latest resolved value of the tracked
        // promises to keep late resolving promises from "cancelling" the state
        // updates performed in this code path.
        // We chain with a void promise to respect `onInput`'s expected return type.
        var _request = _utils.cancelable(runConcurrentSafePromise(collections).then(function() {
            return Promise.resolve();
        }));
        return store.pendingRequests.add(_request);
    }
    setStatus('loading');
    lastStalledId = props.environment.setTimeout(function() {
        setStatus('stalled');
    }, props.stallThreshold); // We track the entire promise chain triggered by `onInput` before mutating
    // the Autocomplete state to make sure that any state manipulation is based on
    // fresh data regardless of when promises individually resolve.
    // We don't track nested promises and only rely on the full chain resolution,
    // meaning we should only ever manipulate the state once this concurrent-safe
    // promise is resolved.
    var request = _utils.cancelable(runConcurrentSafePromise(props.getSources(_objectSpread({
        query: query,
        refresh: refresh,
        state: store.getState()
    }, setters)).then(function(sources) {
        return Promise.all(sources.map(function(source) {
            return Promise.resolve(source.getItems(_objectSpread({
                query: query,
                refresh: refresh,
                state: store.getState()
            }, setters))).then(function(itemsOrDescription) {
                return _resolve.preResolve(itemsOrDescription, source.sourceId);
            });
        })).then(_resolve.resolve).then(function(responses) {
            return _resolve.postResolve(responses, sources);
        }).then(function(collections) {
            return _reshape.reshape({
                collections: collections,
                props: props,
                state: store.getState()
            });
        });
    }))).then(function(collections) {
        var _nextState$isOpen2;
        // Parameters passed to `onInput` could be stale when the following code
        // executes, because `onInput` calls may not resolve in order.
        // If it becomes a problem we'll need to save the last passed parameters.
        // See: https://codesandbox.io/s/agitated-cookies-y290z
        setStatus('idle');
        setCollections(collections);
        var isPanelOpen = props.shouldPanelOpen({
            state: store.getState()
        });
        setIsOpen((_nextState$isOpen2 = nextState.isOpen) !== null && _nextState$isOpen2 !== void 0 ? _nextState$isOpen2 : props.openOnFocus && !query && isPanelOpen || isPanelOpen);
        var highlightedItem = _utils.getActiveItem(store.getState());
        if (store.getState().activeItemId !== null && highlightedItem) {
            var item = highlightedItem.item, itemInputValue = highlightedItem.itemInputValue, itemUrl = highlightedItem.itemUrl, source = highlightedItem.source;
            source.onActive(_objectSpread({
                event: event,
                item: item,
                itemInputValue: itemInputValue,
                itemUrl: itemUrl,
                refresh: refresh,
                source: source,
                state: store.getState()
            }, setters));
        }
    }).finally(function() {
        setStatus('idle');
        if (lastStalledId) props.environment.clearTimeout(lastStalledId);
    });
    return store.pendingRequests.add(request);
}

},{"./reshape":"60vn8","./resolve":"aCMNz","./utils":"gd60Y","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"60vn8":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "reshape", ()=>reshape
);
var _autocompleteShared = require("@algolia/autocomplete-shared");
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        enumerableOnly && (symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        })), keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = null != arguments[i] ? arguments[i] : {};
        i % 2 ? ownKeys(Object(source), !0).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
function reshape(_ref) {
    var collections = _ref.collections, props = _ref.props, state = _ref.state;
    // Sources are grouped by `sourceId` to conveniently pick them via destructuring.
    // Example: `const { recentSearchesPlugin } = sourcesBySourceId`
    var sourcesBySourceId = collections.reduce(function(acc, collection) {
        return _objectSpread(_objectSpread({}, acc), {}, _defineProperty({}, collection.source.sourceId, _objectSpread(_objectSpread({}, collection.source), {}, {
            getItems: function getItems() {
                // We provide the resolved items from the collection to the `reshape` prop.
                return _autocompleteShared.flatten(collection.items);
            }
        })));
    }, {});
    var reshapeSources = props.reshape({
        sources: Object.values(sourcesBySourceId),
        sourcesBySourceId: sourcesBySourceId,
        state: state
    }); // We reconstruct the collections with the items modified by the `reshape` prop.
    return _autocompleteShared.flatten(reshapeSources).filter(Boolean).map(function(source) {
        return {
            source: source,
            items: source.getItems()
        };
    });
}

},{"@algolia/autocomplete-shared":"59T59","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"aCMNz":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "preResolve", ()=>preResolve
);
parcelHelpers.export(exports, "resolve", ()=>resolve
);
parcelHelpers.export(exports, "postResolve", ()=>postResolve
);
var _autocompleteShared = require("@algolia/autocomplete-shared");
var _utils = require("./utils");
function _typeof(obj1) {
    return _typeof = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function(obj) {
        return typeof obj;
    } : function(obj) {
        return obj && "function" == typeof Symbol && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj;
    }, _typeof(obj1);
}
function _toConsumableArray(arr) {
    return _arrayWithoutHoles(arr) || _iterableToArray(arr) || _unsupportedIterableToArray(arr) || _nonIterableSpread();
}
function _nonIterableSpread() {
    throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
}
function _unsupportedIterableToArray(o, minLen) {
    if (!o) return;
    if (typeof o === "string") return _arrayLikeToArray(o, minLen);
    var n = Object.prototype.toString.call(o).slice(8, -1);
    if (n === "Object" && o.constructor) n = o.constructor.name;
    if (n === "Map" || n === "Set") return Array.from(o);
    if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen);
}
function _iterableToArray(iter) {
    if (typeof Symbol !== "undefined" && iter[Symbol.iterator] != null || iter["@@iterator"] != null) return Array.from(iter);
}
function _arrayWithoutHoles(arr) {
    if (Array.isArray(arr)) return _arrayLikeToArray(arr);
}
function _arrayLikeToArray(arr, len) {
    if (len == null || len > arr.length) len = arr.length;
    for(var i = 0, arr2 = new Array(len); i < len; i++)arr2[i] = arr[i];
    return arr2;
}
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        enumerableOnly && (symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        })), keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = null != arguments[i] ? arguments[i] : {};
        i % 2 ? ownKeys(Object(source), !0).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
function isDescription(item) {
    return Boolean(item.execute);
}
function isRequesterDescription(description) {
    return Boolean(description === null || description === void 0 ? void 0 : description.execute);
}
function preResolve(itemsOrDescription, sourceId) {
    if (isRequesterDescription(itemsOrDescription)) return _objectSpread(_objectSpread({}, itemsOrDescription), {}, {
        requests: itemsOrDescription.queries.map(function(query) {
            return {
                query: query,
                sourceId: sourceId,
                transformResponse: itemsOrDescription.transformResponse
            };
        })
    });
    return {
        items: itemsOrDescription,
        sourceId: sourceId
    };
}
function resolve(items1) {
    var packed = items1.reduce(function(acc, current) {
        if (!isDescription(current)) {
            acc.push(current);
            return acc;
        }
        var searchClient = current.searchClient, execute = current.execute, requesterId = current.requesterId, requests = current.requests;
        var container = acc.find(function(item) {
            return isDescription(current) && isDescription(item) && item.searchClient === searchClient && Boolean(requesterId) && item.requesterId === requesterId;
        });
        if (container) {
            var _container$items;
            (_container$items = container.items).push.apply(_container$items, _toConsumableArray(requests));
        } else {
            var request = {
                execute: execute,
                requesterId: requesterId,
                items: requests,
                searchClient: searchClient
            };
            acc.push(request);
        }
        return acc;
    }, []);
    var values = packed.map(function(maybeDescription) {
        if (!isDescription(maybeDescription)) return Promise.resolve(maybeDescription);
        var _ref = maybeDescription, execute = _ref.execute, items = _ref.items, searchClient = _ref.searchClient;
        return execute({
            searchClient: searchClient,
            requests: items
        });
    });
    return Promise.all(values).then(function(responses) {
        return _autocompleteShared.flatten(responses);
    });
}
function postResolve(responses, sources) {
    return sources.map(function(source) {
        var matches = responses.filter(function(response) {
            return response.sourceId === source.sourceId;
        });
        var results = matches.map(function(_ref2) {
            var items = _ref2.items;
            return items;
        });
        var transform = matches[0].transformResponse;
        var items2 = transform ? transform(_utils.mapToAlgoliaResponse(results)) : results;
        _autocompleteShared.invariant(Array.isArray(items2), function() {
            return "The `getItems` function from source \"".concat(source.sourceId, "\" must return an array of items but returned type ").concat(JSON.stringify(_typeof(items2)), ":\n\n").concat(JSON.stringify(_autocompleteShared.decycle(items2), null, 2), ".\n\nSee: https://www.algolia.com/doc/ui-libraries/autocomplete/core-concepts/sources/#param-getitems");
        });
        _autocompleteShared.invariant(items2.every(Boolean), "The `getItems` function from source \"".concat(source.sourceId, "\" must return an array of items but returned ").concat(JSON.stringify(undefined), ".\n\nDid you forget to return items?\n\nSee: https://www.algolia.com/doc/ui-libraries/autocomplete/core-concepts/sources/#param-getitems"));
        return {
            source: source,
            items: items2
        };
    });
}

},{"@algolia/autocomplete-shared":"59T59","./utils":"gd60Y","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"ahwBt":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "onKeyDown", ()=>onKeyDown
);
var _onInput = require("./onInput");
var _utils = require("./utils");
var _excluded = [
    "event",
    "props",
    "refresh",
    "store"
];
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        enumerableOnly && (symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        })), keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = null != arguments[i] ? arguments[i] : {};
        i % 2 ? ownKeys(Object(source), !0).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
function _objectWithoutProperties(source, excluded) {
    if (source == null) return {};
    var target = _objectWithoutPropertiesLoose(source, excluded);
    var key, i;
    if (Object.getOwnPropertySymbols) {
        var sourceSymbolKeys = Object.getOwnPropertySymbols(source);
        for(i = 0; i < sourceSymbolKeys.length; i++){
            key = sourceSymbolKeys[i];
            if (excluded.indexOf(key) >= 0) continue;
            if (!Object.prototype.propertyIsEnumerable.call(source, key)) continue;
            target[key] = source[key];
        }
    }
    return target;
}
function _objectWithoutPropertiesLoose(source, excluded) {
    if (source == null) return {};
    var target = {};
    var sourceKeys = Object.keys(source);
    var key, i;
    for(i = 0; i < sourceKeys.length; i++){
        key = sourceKeys[i];
        if (excluded.indexOf(key) >= 0) continue;
        target[key] = source[key];
    }
    return target;
}
function onKeyDown(_ref) {
    var event = _ref.event, props = _ref.props, refresh = _ref.refresh, store = _ref.store, setters = _objectWithoutProperties(_ref, _excluded);
    if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
        // eslint-disable-next-line no-inner-declarations
        var triggerScrollIntoView = function triggerScrollIntoView() {
            var nodeItem = props.environment.document.getElementById("".concat(props.id, "-item-").concat(store.getState().activeItemId));
            if (nodeItem) {
                if (nodeItem.scrollIntoViewIfNeeded) nodeItem.scrollIntoViewIfNeeded(false);
                else nodeItem.scrollIntoView(false);
            }
        }; // eslint-disable-next-line no-inner-declarations
        var triggerOnActive = function triggerOnActive() {
            var highlightedItem = _utils.getActiveItem(store.getState());
            if (store.getState().activeItemId !== null && highlightedItem) {
                var item = highlightedItem.item, itemInputValue = highlightedItem.itemInputValue, itemUrl = highlightedItem.itemUrl, source = highlightedItem.source;
                source.onActive(_objectSpread({
                    event: event,
                    item: item,
                    itemInputValue: itemInputValue,
                    itemUrl: itemUrl,
                    refresh: refresh,
                    source: source,
                    state: store.getState()
                }, setters));
            }
        }; // Default browser behavior changes the caret placement on ArrowUp and
        // ArrowDown.
        event.preventDefault(); // When re-opening the panel, we need to split the logic to keep the actions
        // synchronized as `onInput` returns a promise.
        if (store.getState().isOpen === false && (props.openOnFocus || Boolean(store.getState().query))) _onInput.onInput(_objectSpread({
            event: event,
            props: props,
            query: store.getState().query,
            refresh: refresh,
            store: store
        }, setters)).then(function() {
            store.dispatch(event.key, {
                nextActiveItemId: props.defaultActiveItemId
            });
            triggerOnActive(); // Since we rely on the DOM, we need to wait for all the micro tasks to
            // finish (which include re-opening the panel) to make sure all the
            // elements are available.
            setTimeout(triggerScrollIntoView, 0);
        });
        else {
            store.dispatch(event.key, {});
            triggerOnActive();
            triggerScrollIntoView();
        }
    } else if (event.key === 'Escape') {
        // This prevents the default browser behavior on `input[type="search"]`
        // from removing the query right away because we first want to close the
        // panel.
        event.preventDefault();
        store.dispatch(event.key, null); // Hitting the `Escape` key signals the end of a user interaction with the
        // autocomplete. At this point, we should ignore any requests that are still
        // pending and could reopen the panel once they resolve, because that would
        // result in an unsolicited UI behavior.
        store.pendingRequests.cancelAll();
    } else if (event.key === 'Enter') {
        // No active item, so we let the browser handle the native `onSubmit` form
        // event.
        if (store.getState().activeItemId === null || store.getState().collections.every(function(collection) {
            return collection.items.length === 0;
        })) return;
         // This prevents the `onSubmit` event to be sent because an item is
        // highlighted.
        event.preventDefault();
        var _ref2 = _utils.getActiveItem(store.getState()), item1 = _ref2.item, itemInputValue1 = _ref2.itemInputValue, itemUrl1 = _ref2.itemUrl, source1 = _ref2.source;
        if (event.metaKey || event.ctrlKey) {
            if (itemUrl1 !== undefined) {
                source1.onSelect(_objectSpread({
                    event: event,
                    item: item1,
                    itemInputValue: itemInputValue1,
                    itemUrl: itemUrl1,
                    refresh: refresh,
                    source: source1,
                    state: store.getState()
                }, setters));
                props.navigator.navigateNewTab({
                    itemUrl: itemUrl1,
                    item: item1,
                    state: store.getState()
                });
            }
        } else if (event.shiftKey) {
            if (itemUrl1 !== undefined) {
                source1.onSelect(_objectSpread({
                    event: event,
                    item: item1,
                    itemInputValue: itemInputValue1,
                    itemUrl: itemUrl1,
                    refresh: refresh,
                    source: source1,
                    state: store.getState()
                }, setters));
                props.navigator.navigateNewWindow({
                    itemUrl: itemUrl1,
                    item: item1,
                    state: store.getState()
                });
            }
        } else if (event.altKey) ;
        else {
            if (itemUrl1 !== undefined) {
                source1.onSelect(_objectSpread({
                    event: event,
                    item: item1,
                    itemInputValue: itemInputValue1,
                    itemUrl: itemUrl1,
                    refresh: refresh,
                    source: source1,
                    state: store.getState()
                }, setters));
                props.navigator.navigate({
                    itemUrl: itemUrl1,
                    item: item1,
                    state: store.getState()
                });
                return;
            }
            _onInput.onInput(_objectSpread({
                event: event,
                nextState: {
                    isOpen: false
                },
                props: props,
                query: itemInputValue1,
                refresh: refresh,
                store: store
            }, setters)).then(function() {
                source1.onSelect(_objectSpread({
                    event: event,
                    item: item1,
                    itemInputValue: itemInputValue1,
                    itemUrl: itemUrl1,
                    refresh: refresh,
                    source: source1,
                    state: store.getState()
                }, setters));
            });
        }
    }
}

},{"./onInput":"6DlJz","./utils":"gd60Y","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"66SZT":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "getMetadata", ()=>getMetadata
);
parcelHelpers.export(exports, "injectMetadata", ()=>injectMetadata
);
var _autocompleteShared = require("@algolia/autocomplete-shared");
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        enumerableOnly && (symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        })), keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = null != arguments[i] ? arguments[i] : {};
        i % 2 ? ownKeys(Object(source), !0).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
function getMetadata(_ref) {
    var _, _options$__autocomple, _options$__autocomple2, _options$__autocomple3;
    var plugins = _ref.plugins, options = _ref.options;
    var optionsKey = (_ = (((_options$__autocomple = options.__autocomplete_metadata) === null || _options$__autocomple === void 0 ? void 0 : _options$__autocomple.userAgents) || [])[0]) === null || _ === void 0 ? void 0 : _.segment;
    var extraOptions = optionsKey ? _defineProperty({}, optionsKey, Object.keys(((_options$__autocomple2 = options.__autocomplete_metadata) === null || _options$__autocomple2 === void 0 ? void 0 : _options$__autocomple2.options) || {})) : {};
    return {
        plugins: plugins.map(function(plugin) {
            return {
                name: plugin.name,
                options: Object.keys(plugin.__autocomplete_pluginOptions || [])
            };
        }),
        options: _objectSpread({
            'autocomplete-core': Object.keys(options)
        }, extraOptions),
        ua: _autocompleteShared.userAgents.concat(((_options$__autocomple3 = options.__autocomplete_metadata) === null || _options$__autocomple3 === void 0 ? void 0 : _options$__autocomple3.userAgents) || [])
    };
}
function injectMetadata(_ref3) {
    var _environment$navigato;
    var metadata = _ref3.metadata, environment = _ref3.environment;
    var isMetadataEnabled = (_environment$navigato = environment.navigator) === null || _environment$navigato === void 0 ? void 0 : _environment$navigato.userAgent.includes('Algolia Crawler');
    if (isMetadataEnabled) {
        var metadataContainer = environment.document.createElement('meta');
        var headRef = environment.document.querySelector('head');
        metadataContainer.name = 'algolia:metadata';
        setTimeout(function() {
            metadataContainer.content = JSON.stringify(metadata);
            headRef.appendChild(metadataContainer);
        }, 0);
    }
}

},{"@algolia/autocomplete-shared":"59T59","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"iw5Pd":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "stateReducer", ()=>stateReducer
);
var _autocompleteShared = require("@algolia/autocomplete-shared");
var _getCompletion = require("./getCompletion");
var _utils = require("./utils");
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        enumerableOnly && (symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        })), keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = null != arguments[i] ? arguments[i] : {};
        i % 2 ? ownKeys(Object(source), !0).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
var stateReducer = function stateReducer(state, action) {
    switch(action.type){
        case 'setActiveItemId':
            return _objectSpread(_objectSpread({}, state), {}, {
                activeItemId: action.payload
            });
        case 'setQuery':
            return _objectSpread(_objectSpread({}, state), {}, {
                query: action.payload,
                completion: null
            });
        case 'setCollections':
            return _objectSpread(_objectSpread({}, state), {}, {
                collections: action.payload
            });
        case 'setIsOpen':
            return _objectSpread(_objectSpread({}, state), {}, {
                isOpen: action.payload
            });
        case 'setStatus':
            return _objectSpread(_objectSpread({}, state), {}, {
                status: action.payload
            });
        case 'setContext':
            return _objectSpread(_objectSpread({}, state), {}, {
                context: _objectSpread(_objectSpread({}, state.context), action.payload)
            });
        case 'ArrowDown':
            var nextState = _objectSpread(_objectSpread({}, state), {}, {
                activeItemId: action.payload.hasOwnProperty('nextActiveItemId') ? action.payload.nextActiveItemId : _utils.getNextActiveItemId(1, state.activeItemId, _autocompleteShared.getItemsCount(state), action.props.defaultActiveItemId)
            });
            return _objectSpread(_objectSpread({}, nextState), {}, {
                completion: _getCompletion.getCompletion({
                    state: nextState
                })
            });
        case 'ArrowUp':
            var _nextState = _objectSpread(_objectSpread({}, state), {}, {
                activeItemId: _utils.getNextActiveItemId(-1, state.activeItemId, _autocompleteShared.getItemsCount(state), action.props.defaultActiveItemId)
            });
            return _objectSpread(_objectSpread({}, _nextState), {}, {
                completion: _getCompletion.getCompletion({
                    state: _nextState
                })
            });
        case 'Escape':
            if (state.isOpen) return _objectSpread(_objectSpread({}, state), {}, {
                activeItemId: null,
                isOpen: false,
                completion: null
            });
            return _objectSpread(_objectSpread({}, state), {}, {
                activeItemId: null,
                query: '',
                status: 'idle',
                collections: []
            });
        case 'submit':
            return _objectSpread(_objectSpread({}, state), {}, {
                activeItemId: null,
                isOpen: false,
                status: 'idle'
            });
        case 'reset':
            return _objectSpread(_objectSpread({}, state), {}, {
                activeItemId: // we need to restore the highlighted index to the defaultActiveItemId. (DocSearch use-case)
                // Since we close the panel when openOnFocus=false
                // we lose track of the highlighted index. (Query-suggestions use-case)
                action.props.openOnFocus === true ? action.props.defaultActiveItemId : null,
                status: 'idle',
                query: ''
            });
        case 'focus':
            return _objectSpread(_objectSpread({}, state), {}, {
                activeItemId: action.props.defaultActiveItemId,
                isOpen: (action.props.openOnFocus || Boolean(state.query)) && action.props.shouldPanelOpen({
                    state: state
                })
            });
        case 'blur':
            if (action.props.debug) return state;
            return _objectSpread(_objectSpread({}, state), {}, {
                isOpen: false,
                activeItemId: null
            });
        case 'mousemove':
            return _objectSpread(_objectSpread({}, state), {}, {
                activeItemId: action.payload
            });
        case 'mouseleave':
            return _objectSpread(_objectSpread({}, state), {}, {
                activeItemId: action.props.defaultActiveItemId
            });
        default:
            _autocompleteShared.invariant(false, "The reducer action ".concat(JSON.stringify(action.type), " is not supported."));
            return state;
    }
};

},{"@algolia/autocomplete-shared":"59T59","./getCompletion":"faKMA","./utils":"gd60Y","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"faKMA":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "getCompletion", ()=>getCompletion
);
var _utils = require("./utils");
function getCompletion(_ref) {
    var _getActiveItem;
    var state = _ref.state;
    if (state.isOpen === false || state.activeItemId === null) return null;
    return ((_getActiveItem = _utils.getActiveItem(state)) === null || _getActiveItem === void 0 ? void 0 : _getActiveItem.itemInputValue) || null;
}

},{"./utils":"gd60Y","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"79Luq":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _autocompleteApi = require("./AutocompleteApi");
parcelHelpers.exportAll(_autocompleteApi, exports);
var _autocompleteCollection = require("./AutocompleteCollection");
parcelHelpers.exportAll(_autocompleteCollection, exports);
var _autocompleteContext = require("./AutocompleteContext");
parcelHelpers.exportAll(_autocompleteContext, exports);
var _autocompleteEnvironment = require("./AutocompleteEnvironment");
parcelHelpers.exportAll(_autocompleteEnvironment, exports);
var _autocompleteOptions = require("./AutocompleteOptions");
parcelHelpers.exportAll(_autocompleteOptions, exports);
var _autocompleteSource = require("./AutocompleteSource");
parcelHelpers.exportAll(_autocompleteSource, exports);
var _autocompletePropGetters = require("./AutocompletePropGetters");
parcelHelpers.exportAll(_autocompletePropGetters, exports);
var _autocompletePlugin = require("./AutocompletePlugin");
parcelHelpers.exportAll(_autocompletePlugin, exports);
var _autocompleteReshape = require("./AutocompleteReshape");
parcelHelpers.exportAll(_autocompleteReshape, exports);
var _autocompleteSetters = require("./AutocompleteSetters");
parcelHelpers.exportAll(_autocompleteSetters, exports);
var _autocompleteState = require("./AutocompleteState");
parcelHelpers.exportAll(_autocompleteState, exports);
var _autocompleteStore = require("./AutocompleteStore");
parcelHelpers.exportAll(_autocompleteStore, exports);
var _autocompleteSubscribers = require("./AutocompleteSubscribers");
parcelHelpers.exportAll(_autocompleteSubscribers, exports);

},{"./AutocompleteApi":"bZskC","./AutocompleteCollection":"9z6Jx","./AutocompleteContext":"jsDWg","./AutocompleteEnvironment":"50zbt","./AutocompleteOptions":"1fpkb","./AutocompleteSource":"3MW3B","./AutocompletePropGetters":"byZ8W","./AutocompletePlugin":"cr6ih","./AutocompleteReshape":"gKdoQ","./AutocompleteSetters":"3od6P","./AutocompleteState":"cqghE","./AutocompleteStore":"aCF9w","./AutocompleteSubscribers":"jtnxo","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"bZskC":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"9z6Jx":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"jsDWg":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"50zbt":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"1fpkb":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"3MW3B":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"byZ8W":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"cr6ih":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"gKdoQ":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"3od6P":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"cqghE":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"aCF9w":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"jtnxo":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"58er4":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var n = function(t1, s, r, e) {
    var u;
    s[0] = 0;
    for(var h = 1; h < s.length; h++){
        var p = s[h++], a = s[h] ? (s[0] |= p ? 1 : 2, r[s[h++]]) : s[++h];
        3 === p ? e[0] = a : 4 === p ? e[1] = Object.assign(e[1] || {}, a) : 5 === p ? (e[1] = e[1] || {})[s[++h]] = a : 6 === p ? e[1][s[++h]] += a + "" : p ? (u = t1.apply(a, n(t1, a, r, [
            "",
            null
        ])), e.push(u), a[0] ? s[0] |= 2 : (s[h - 2] = 0, s[h] = u)) : e.push(a);
    }
    return e;
}, t = new Map;
exports.default = function(s1) {
    var r1 = t.get(this);
    return r1 || (r1 = new Map, t.set(this, r1)), (r1 = n(this, r1.get(s1) || (r1.set(s1, r1 = function(n1) {
        for(var t2, s, r = 1, e = "", u = "", h = [
            0
        ], p = function(n2) {
            1 === r && (n2 || (e = e.replace(/^\s*\n\s*|\s*\n\s*$/g, ""))) ? h.push(0, n2, e) : 3 === r && (n2 || e) ? (h.push(3, n2, e), r = 2) : 2 === r && "..." === e && n2 ? h.push(4, n2, 0) : 2 === r && e && !n2 ? h.push(5, 0, !0, e) : r >= 5 && ((e || !n2 && 5 === r) && (h.push(r, 0, e, s), r = 6), n2 && (h.push(r, n2, 0, s), r = 6)), e = "";
        }, a = 0; a < n1.length; a++){
            a && (1 === r && p(), p(a));
            for(var l = 0; l < n1[a].length; l++)t2 = n1[a][l], 1 === r ? "<" === t2 ? (p(), h = [
                h
            ], r = 3) : e += t2 : 4 === r ? "--" === e && ">" === t2 ? (r = 1, e = "") : e = t2 + e[0] : u ? t2 === u ? u = "" : e += t2 : '"' === t2 || "'" === t2 ? u = t2 : ">" === t2 ? (p(), r = 1) : r && ("=" === t2 ? (r = 5, s = e, e = "") : "/" === t2 && (r < 5 || ">" === n1[a][l + 1]) ? (p(), 3 === r && (h = h[0]), r = h, (h = h[0]).push(2, 0, r), r = 0) : " " === t2 || "\t" === t2 || "\n" === t2 || "\r" === t2 ? (p(), r = 2) : e += t2), 3 === r && "!--" === e && (r = 4, h = h[0]);
        }
        return p(), h;
    }(s1)), r1), arguments, [])).length > 1 ? r1 : r1[0];
};

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"fEfqR":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "createAutocompleteDom", ()=>createAutocompleteDom
);
var _elements = require("./elements");
var _getCreateDomElement = require("./getCreateDomElement");
var _utils = require("./utils");
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        enumerableOnly && (symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        })), keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = null != arguments[i] ? arguments[i] : {};
        i % 2 ? ownKeys(Object(source), !0).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
function createAutocompleteDom(_ref) {
    var autocomplete = _ref.autocomplete, autocompleteScopeApi = _ref.autocompleteScopeApi, classNames = _ref.classNames, environment = _ref.environment, isDetached = _ref.isDetached, _ref$placeholder = _ref.placeholder, placeholder = _ref$placeholder === void 0 ? 'Search' : _ref$placeholder, propGetters = _ref.propGetters, setIsModalOpen = _ref.setIsModalOpen, state = _ref.state, translations = _ref.translations;
    var createDomElement = _getCreateDomElement.getCreateDomElement(environment);
    var rootProps = propGetters.getRootProps(_objectSpread({
        state: state,
        props: autocomplete.getRootProps({})
    }, autocompleteScopeApi));
    var root = createDomElement('div', _objectSpread({
        class: classNames.root
    }, rootProps));
    var detachedContainer = createDomElement('div', {
        class: classNames.detachedContainer,
        onMouseDown: function onMouseDown(event) {
            event.stopPropagation();
        }
    });
    var detachedOverlay = createDomElement('div', {
        class: classNames.detachedOverlay,
        children: [
            detachedContainer
        ],
        onMouseDown: function onMouseDown() {
            setIsModalOpen(false);
            autocomplete.setIsOpen(false);
        }
    });
    var labelProps = propGetters.getLabelProps(_objectSpread({
        state: state,
        props: autocomplete.getLabelProps({})
    }, autocompleteScopeApi));
    var submitButton = createDomElement('button', {
        class: classNames.submitButton,
        type: 'submit',
        title: translations.submitButtonTitle,
        children: [
            _elements.SearchIcon({
                environment: environment
            })
        ]
    });
    var label = createDomElement('label', _objectSpread({
        class: classNames.label,
        children: [
            submitButton
        ]
    }, labelProps));
    var clearButton = createDomElement('button', {
        class: classNames.clearButton,
        type: 'reset',
        title: translations.clearButtonTitle,
        children: [
            _elements.ClearIcon({
                environment: environment
            })
        ]
    });
    var loadingIndicator = createDomElement('div', {
        class: classNames.loadingIndicator,
        children: [
            _elements.LoadingIcon({
                environment: environment
            })
        ]
    });
    var input = _elements.Input({
        class: classNames.input,
        environment: environment,
        state: state,
        getInputProps: propGetters.getInputProps,
        getInputPropsCore: autocomplete.getInputProps,
        autocompleteScopeApi: autocompleteScopeApi,
        onDetachedEscape: isDetached ? function() {
            autocomplete.setIsOpen(false);
            setIsModalOpen(false);
        } : undefined
    });
    var inputWrapperPrefix = createDomElement('div', {
        class: classNames.inputWrapperPrefix,
        children: [
            label,
            loadingIndicator
        ]
    });
    var inputWrapperSuffix = createDomElement('div', {
        class: classNames.inputWrapperSuffix,
        children: [
            clearButton
        ]
    });
    var inputWrapper = createDomElement('div', {
        class: classNames.inputWrapper,
        children: [
            input
        ]
    });
    var formProps = propGetters.getFormProps(_objectSpread({
        state: state,
        props: autocomplete.getFormProps({
            inputElement: input
        })
    }, autocompleteScopeApi));
    var form = createDomElement('form', _objectSpread({
        class: classNames.form,
        children: [
            inputWrapperPrefix,
            inputWrapper,
            inputWrapperSuffix
        ]
    }, formProps));
    var panelProps = propGetters.getPanelProps(_objectSpread({
        state: state,
        props: autocomplete.getPanelProps({})
    }, autocompleteScopeApi));
    var panel = createDomElement('div', _objectSpread({
        class: classNames.panel
    }, panelProps));
    if (isDetached) {
        var detachedSearchButtonIcon = createDomElement('div', {
            class: classNames.detachedSearchButtonIcon,
            children: [
                _elements.SearchIcon({
                    environment: environment
                })
            ]
        });
        var detachedSearchButtonPlaceholder = createDomElement('div', {
            class: classNames.detachedSearchButtonPlaceholder,
            textContent: placeholder
        });
        var detachedSearchButton = createDomElement('button', {
            type: 'button',
            class: classNames.detachedSearchButton,
            onClick: function onClick() {
                setIsModalOpen(true);
            },
            children: [
                detachedSearchButtonIcon,
                detachedSearchButtonPlaceholder
            ]
        });
        var detachedCancelButton = createDomElement('button', {
            type: 'button',
            class: classNames.detachedCancelButton,
            textContent: translations.detachedCancelButtonText,
            // Prevent `onTouchStart` from closing the panel
            // since it should be initiated by `onClick` only
            onTouchStart: function onTouchStart(event) {
                event.stopPropagation();
            },
            onClick: function onClick() {
                autocomplete.setIsOpen(false);
                setIsModalOpen(false);
            }
        });
        var detachedFormContainer = createDomElement('div', {
            class: classNames.detachedFormContainer,
            children: [
                form,
                detachedCancelButton
            ]
        });
        detachedContainer.appendChild(detachedFormContainer);
        root.appendChild(detachedSearchButton);
    } else root.appendChild(form);
    return {
        detachedContainer: detachedContainer,
        detachedOverlay: detachedOverlay,
        inputWrapper: inputWrapper,
        input: input,
        root: root,
        form: form,
        label: label,
        submitButton: submitButton,
        clearButton: clearButton,
        loadingIndicator: loadingIndicator,
        panel: panel
    };
}

},{"./elements":"hT37i","./getCreateDomElement":"dupN1","./utils":"fKU1x","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"hT37i":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _clearIcon = require("./ClearIcon");
parcelHelpers.exportAll(_clearIcon, exports);
var _input = require("./Input");
parcelHelpers.exportAll(_input, exports);
var _loadingIcon = require("./LoadingIcon");
parcelHelpers.exportAll(_loadingIcon, exports);
var _searchIcon = require("./SearchIcon");
parcelHelpers.exportAll(_searchIcon, exports);

},{"./ClearIcon":"dWjga","./Input":"lyL9J","./LoadingIcon":"f4T3P","./SearchIcon":"bK1wB","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"dWjga":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "ClearIcon", ()=>ClearIcon
);
var ClearIcon = function ClearIcon(_ref) {
    var environment = _ref.environment;
    var element = environment.document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    element.setAttribute('class', 'aa-ClearIcon');
    element.setAttribute('viewBox', '0 0 24 24');
    element.setAttribute('width', '18');
    element.setAttribute('height', '18');
    element.setAttribute('fill', 'currentColor');
    var path = environment.document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', 'M5.293 6.707l5.293 5.293-5.293 5.293c-0.391 0.391-0.391 1.024 0 1.414s1.024 0.391 1.414 0l5.293-5.293 5.293 5.293c0.391 0.391 1.024 0.391 1.414 0s0.391-1.024 0-1.414l-5.293-5.293 5.293-5.293c0.391-0.391 0.391-1.024 0-1.414s-1.024-0.391-1.414 0l-5.293 5.293-5.293-5.293c-0.391-0.391-1.024-0.391-1.414 0s-0.391 1.024 0 1.414z');
    element.appendChild(path);
    return element;
};

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"lyL9J":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "Input", ()=>Input
);
var _getCreateDomElement = require("../getCreateDomElement");
var _utils = require("../utils");
var _excluded = [
    "autocompleteScopeApi",
    "environment",
    "classNames",
    "getInputProps",
    "getInputPropsCore",
    "onDetachedEscape",
    "state"
];
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        enumerableOnly && (symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        })), keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = null != arguments[i] ? arguments[i] : {};
        i % 2 ? ownKeys(Object(source), !0).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
function _objectWithoutProperties(source, excluded) {
    if (source == null) return {};
    var target = _objectWithoutPropertiesLoose(source, excluded);
    var key, i;
    if (Object.getOwnPropertySymbols) {
        var sourceSymbolKeys = Object.getOwnPropertySymbols(source);
        for(i = 0; i < sourceSymbolKeys.length; i++){
            key = sourceSymbolKeys[i];
            if (excluded.indexOf(key) >= 0) continue;
            if (!Object.prototype.propertyIsEnumerable.call(source, key)) continue;
            target[key] = source[key];
        }
    }
    return target;
}
function _objectWithoutPropertiesLoose(source, excluded) {
    if (source == null) return {};
    var target = {};
    var sourceKeys = Object.keys(source);
    var key, i;
    for(i = 0; i < sourceKeys.length; i++){
        key = sourceKeys[i];
        if (excluded.indexOf(key) >= 0) continue;
        target[key] = source[key];
    }
    return target;
}
var Input = function Input(_ref) {
    var autocompleteScopeApi = _ref.autocompleteScopeApi, environment = _ref.environment, classNames = _ref.classNames, getInputProps = _ref.getInputProps, getInputPropsCore = _ref.getInputPropsCore, onDetachedEscape = _ref.onDetachedEscape, state = _ref.state, props = _objectWithoutProperties(_ref, _excluded);
    var createDomElement = _getCreateDomElement.getCreateDomElement(environment);
    var element = createDomElement('input', props);
    var inputProps = getInputProps(_objectSpread({
        state: state,
        props: getInputPropsCore({
            inputElement: element
        }),
        inputElement: element
    }, autocompleteScopeApi));
    _utils.setProperties(element, _objectSpread(_objectSpread({}, inputProps), {}, {
        onKeyDown: function onKeyDown(event) {
            if (onDetachedEscape && event.key === 'Escape') {
                event.preventDefault();
                onDetachedEscape();
                return;
            }
            inputProps.onKeyDown(event);
        }
    }));
    return element;
};

},{"../getCreateDomElement":"dupN1","../utils":"fKU1x","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"dupN1":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "getCreateDomElement", ()=>getCreateDomElement
);
var _utils = require("./utils");
var _excluded = [
    "children"
];
function _toConsumableArray(arr) {
    return _arrayWithoutHoles(arr) || _iterableToArray(arr) || _unsupportedIterableToArray(arr) || _nonIterableSpread();
}
function _nonIterableSpread() {
    throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
}
function _unsupportedIterableToArray(o, minLen) {
    if (!o) return;
    if (typeof o === "string") return _arrayLikeToArray(o, minLen);
    var n = Object.prototype.toString.call(o).slice(8, -1);
    if (n === "Object" && o.constructor) n = o.constructor.name;
    if (n === "Map" || n === "Set") return Array.from(o);
    if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen);
}
function _iterableToArray(iter) {
    if (typeof Symbol !== "undefined" && iter[Symbol.iterator] != null || iter["@@iterator"] != null) return Array.from(iter);
}
function _arrayWithoutHoles(arr) {
    if (Array.isArray(arr)) return _arrayLikeToArray(arr);
}
function _arrayLikeToArray(arr, len) {
    if (len == null || len > arr.length) len = arr.length;
    for(var i = 0, arr2 = new Array(len); i < len; i++)arr2[i] = arr[i];
    return arr2;
}
function _objectWithoutProperties(source, excluded) {
    if (source == null) return {};
    var target = _objectWithoutPropertiesLoose(source, excluded);
    var key, i;
    if (Object.getOwnPropertySymbols) {
        var sourceSymbolKeys = Object.getOwnPropertySymbols(source);
        for(i = 0; i < sourceSymbolKeys.length; i++){
            key = sourceSymbolKeys[i];
            if (excluded.indexOf(key) >= 0) continue;
            if (!Object.prototype.propertyIsEnumerable.call(source, key)) continue;
            target[key] = source[key];
        }
    }
    return target;
}
function _objectWithoutPropertiesLoose(source, excluded) {
    if (source == null) return {};
    var target = {};
    var sourceKeys = Object.keys(source);
    var key, i;
    for(i = 0; i < sourceKeys.length; i++){
        key = sourceKeys[i];
        if (excluded.indexOf(key) >= 0) continue;
        target[key] = source[key];
    }
    return target;
}
function getCreateDomElement(environment) {
    return function createDomElement(tagName, _ref) {
        var _ref$children = _ref.children, children = _ref$children === void 0 ? [] : _ref$children, props = _objectWithoutProperties(_ref, _excluded);
        var element = environment.document.createElement(tagName);
        _utils.setProperties(element, props);
        element.append.apply(element, _toConsumableArray(children));
        return element;
    };
}

},{"./utils":"fKU1x","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"fKU1x":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _getHTMLElement = require("./getHTMLElement");
parcelHelpers.exportAll(_getHTMLElement, exports);
var _mergeClassNames = require("./mergeClassNames");
parcelHelpers.exportAll(_mergeClassNames, exports);
var _mergeDeep = require("./mergeDeep");
parcelHelpers.exportAll(_mergeDeep, exports);
var _pickBy = require("./pickBy");
parcelHelpers.exportAll(_pickBy, exports);
var _setProperties = require("./setProperties");
parcelHelpers.exportAll(_setProperties, exports);

},{"./getHTMLElement":"5LR0S","./mergeClassNames":"ioFOf","./mergeDeep":"dB2gA","./pickBy":"9Qraf","./setProperties":"8UKdL","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"5LR0S":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "getHTMLElement", ()=>getHTMLElement
);
var _autocompleteShared = require("@algolia/autocomplete-shared");
function getHTMLElement(environment, value) {
    if (typeof value === 'string') {
        var element = environment.document.querySelector(value);
        _autocompleteShared.invariant(element !== null, "The element ".concat(JSON.stringify(value), " is not in the document."));
        return element;
    }
    return value;
}

},{"@algolia/autocomplete-shared":"59T59","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"ioFOf":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "mergeClassNames", ()=>mergeClassNames
);
function mergeClassNames() {
    for(var _len = arguments.length, values = new Array(_len), _key = 0; _key < _len; _key++)values[_key] = arguments[_key];
    return values.reduce(function(acc, current) {
        Object.keys(current).forEach(function(key) {
            var accValue = acc[key];
            var currentValue = current[key];
            if (accValue !== currentValue) acc[key] = [
                accValue,
                currentValue
            ].filter(Boolean).join(' ');
        });
        return acc;
    }, {});
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"dB2gA":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "mergeDeep", ()=>mergeDeep
);
function _toConsumableArray(arr) {
    return _arrayWithoutHoles(arr) || _iterableToArray(arr) || _unsupportedIterableToArray(arr) || _nonIterableSpread();
}
function _nonIterableSpread() {
    throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
}
function _unsupportedIterableToArray(o, minLen) {
    if (!o) return;
    if (typeof o === "string") return _arrayLikeToArray(o, minLen);
    var n = Object.prototype.toString.call(o).slice(8, -1);
    if (n === "Object" && o.constructor) n = o.constructor.name;
    if (n === "Map" || n === "Set") return Array.from(o);
    if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen);
}
function _iterableToArray(iter) {
    if (typeof Symbol !== "undefined" && iter[Symbol.iterator] != null || iter["@@iterator"] != null) return Array.from(iter);
}
function _arrayWithoutHoles(arr) {
    if (Array.isArray(arr)) return _arrayLikeToArray(arr);
}
function _arrayLikeToArray(arr, len) {
    if (len == null || len > arr.length) len = arr.length;
    for(var i = 0, arr2 = new Array(len); i < len; i++)arr2[i] = arr[i];
    return arr2;
}
function _typeof(obj1) {
    return _typeof = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function(obj) {
        return typeof obj;
    } : function(obj) {
        return obj && "function" == typeof Symbol && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj;
    }, _typeof(obj1);
}
var isPlainObject = function isPlainObject(value) {
    return value && _typeof(value) === 'object' && Object.prototype.toString.call(value) === '[object Object]';
};
function mergeDeep() {
    for(var _len = arguments.length, values = new Array(_len), _key = 0; _key < _len; _key++)values[_key] = arguments[_key];
    return values.reduce(function(acc, current) {
        Object.keys(current).forEach(function(key) {
            var accValue = acc[key];
            var currentValue = current[key];
            if (Array.isArray(accValue) && Array.isArray(currentValue)) acc[key] = accValue.concat.apply(accValue, _toConsumableArray(currentValue));
            else if (isPlainObject(accValue) && isPlainObject(currentValue)) acc[key] = mergeDeep(accValue, currentValue);
            else acc[key] = currentValue;
        });
        return acc;
    }, {});
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"9Qraf":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "pickBy", ()=>pickBy
);
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        enumerableOnly && (symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        })), keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = null != arguments[i] ? arguments[i] : {};
        i % 2 ? ownKeys(Object(source), !0).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
function _slicedToArray(arr, i) {
    return _arrayWithHoles(arr) || _iterableToArrayLimit(arr, i) || _unsupportedIterableToArray(arr, i) || _nonIterableRest();
}
function _nonIterableRest() {
    throw new TypeError("Invalid attempt to destructure non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
}
function _unsupportedIterableToArray(o, minLen) {
    if (!o) return;
    if (typeof o === "string") return _arrayLikeToArray(o, minLen);
    var n = Object.prototype.toString.call(o).slice(8, -1);
    if (n === "Object" && o.constructor) n = o.constructor.name;
    if (n === "Map" || n === "Set") return Array.from(o);
    if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen);
}
function _arrayLikeToArray(arr, len) {
    if (len == null || len > arr.length) len = arr.length;
    for(var i = 0, arr2 = new Array(len); i < len; i++)arr2[i] = arr[i];
    return arr2;
}
function _iterableToArrayLimit(arr, i) {
    var _i = arr == null ? null : typeof Symbol !== "undefined" && arr[Symbol.iterator] || arr["@@iterator"];
    if (_i == null) return;
    var _arr = [];
    var _n = true;
    var _d = false;
    var _s, _e;
    try {
        for(_i = _i.call(arr); !(_n = (_s = _i.next()).done); _n = true){
            _arr.push(_s.value);
            if (i && _arr.length === i) break;
        }
    } catch (err) {
        _d = true;
        _e = err;
    } finally{
        try {
            if (!_n && _i["return"] != null) _i["return"]();
        } finally{
            if (_d) throw _e;
        }
    }
    return _arr;
}
function _arrayWithHoles(arr) {
    if (Array.isArray(arr)) return arr;
}
function pickBy(obj, predicate) {
    return Object.entries(obj).reduce(function(acc, _ref) {
        var _ref2 = _slicedToArray(_ref, 2), key = _ref2[0], value = _ref2[1];
        if (predicate({
            key: key,
            value: value
        })) return _objectSpread(_objectSpread({}, acc), {}, _defineProperty({}, key, value));
        return acc;
    }, {});
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"8UKdL":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/**
 * Set a property value on a DOM node
 */ parcelHelpers.export(exports, "setProperty", ()=>setProperty
);
parcelHelpers.export(exports, "setProperties", ()=>setProperties
);
parcelHelpers.export(exports, "setPropertiesWithoutEvents", ()=>setPropertiesWithoutEvents
);
/* eslint-disable */ /**
 * Touch-specific event aliases
 *
 * See https://w3c.github.io/touch-events/#extensions-to-the-globaleventhandlers-mixin
 */ var TOUCH_EVENTS_ALIASES = [
    'ontouchstart',
    'ontouchend',
    'ontouchmove',
    'ontouchcancel'
];
/*
 * Taken from Preact
 *
 * See https://github.com/preactjs/preact/blob/6ab49d9020740127577bf4af66bf63f4af7f9fee/src/diff/props.js#L58-L151
 */ function setStyle(style, key, value) {
    if (value === null) style[key] = '';
    else if (typeof value !== 'number') style[key] = value;
    else style[key] = value + 'px';
}
/**
 * Proxy an event to hooked event handlers
 */ function eventProxy(event) {
    this._listeners[event.type](event);
}
function setProperty(dom, name, value) {
    var useCapture;
    var nameLower;
    var oldValue = dom[name];
    if (name === 'style') {
        if (typeof value == 'string') dom.style = value;
        else if (value === null) dom.style = '';
        else {
            for(name in value)if (!oldValue || value[name] !== oldValue[name]) setStyle(dom.style, name, value[name]);
        }
    } else if (name[0] === 'o' && name[1] === 'n') {
        useCapture = name !== (name = name.replace(/Capture$/, ''));
        nameLower = name.toLowerCase();
        if (nameLower in dom || TOUCH_EVENTS_ALIASES.includes(nameLower)) name = nameLower;
        name = name.slice(2);
        if (!dom._listeners) dom._listeners = {};
        dom._listeners[name] = value;
        if (value) {
            if (!oldValue) dom.addEventListener(name, eventProxy, useCapture);
        } else dom.removeEventListener(name, eventProxy, useCapture);
    } else if (name !== 'list' && name !== 'tagName' && // setAttribute
    name !== 'form' && name !== 'type' && name !== 'size' && name !== 'download' && name !== 'href' && name in dom) dom[name] = value == null ? '' : value;
    else if (typeof value != 'function' && name !== 'dangerouslySetInnerHTML') {
        if (value == null || value === false && // The value `false` is different from the attribute not
        // existing on the DOM, so we can't remove it. For non-boolean
        // ARIA-attributes we could treat false as a removal, but the
        // amount of exceptions would cost us too many bytes. On top of
        // that other VDOM frameworks also always stringify `false`.
        !/^ar/.test(name)) dom.removeAttribute(name);
        else dom.setAttribute(name, value);
    }
}
function getNormalizedName(name) {
    switch(name){
        case 'onChange':
            return 'onInput';
        default:
            return name;
    }
}
function setProperties(dom, props) {
    for(var name in props)setProperty(dom, getNormalizedName(name), props[name]);
}
function setPropertiesWithoutEvents(dom, props) {
    for(var name in props)if (!(name[0] === 'o' && name[1] === 'n')) setProperty(dom, getNormalizedName(name), props[name]);
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"f4T3P":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "LoadingIcon", ()=>LoadingIcon
);
var LoadingIcon = function LoadingIcon(_ref) {
    var environment = _ref.environment;
    var element = environment.document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    element.setAttribute('class', 'aa-LoadingIcon');
    element.setAttribute('viewBox', '0 0 100 100');
    element.setAttribute('width', '20');
    element.setAttribute('height', '20');
    element.innerHTML = "<circle\n  cx=\"50\"\n  cy=\"50\"\n  fill=\"none\"\n  r=\"35\"\n  stroke=\"currentColor\"\n  stroke-dasharray=\"164.93361431346415 56.97787143782138\"\n  stroke-width=\"6\"\n>\n  <animateTransform\n    attributeName=\"transform\"\n    type=\"rotate\"\n    repeatCount=\"indefinite\"\n    dur=\"1s\"\n    values=\"0 50 50;90 50 50;180 50 50;360 50 50\"\n    keyTimes=\"0;0.40;0.65;1\"\n  />\n</circle>";
    return element;
};

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"bK1wB":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "SearchIcon", ()=>SearchIcon
);
var SearchIcon = function SearchIcon(_ref) {
    var environment = _ref.environment;
    var element = environment.document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    element.setAttribute('class', 'aa-SubmitIcon');
    element.setAttribute('viewBox', '0 0 24 24');
    element.setAttribute('width', '20');
    element.setAttribute('height', '20');
    element.setAttribute('fill', 'currentColor');
    var path = environment.document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', 'M16.041 15.856c-0.034 0.026-0.067 0.055-0.099 0.087s-0.060 0.064-0.087 0.099c-1.258 1.213-2.969 1.958-4.855 1.958-1.933 0-3.682-0.782-4.95-2.050s-2.050-3.017-2.050-4.95 0.782-3.682 2.050-4.95 3.017-2.050 4.95-2.050 3.682 0.782 4.95 2.050 2.050 3.017 2.050 4.95c0 1.886-0.745 3.597-1.959 4.856zM21.707 20.293l-3.675-3.675c1.231-1.54 1.968-3.493 1.968-5.618 0-2.485-1.008-4.736-2.636-6.364s-3.879-2.636-6.364-2.636-4.736 1.008-6.364 2.636-2.636 3.879-2.636 6.364 1.008 4.736 2.636 6.364 3.879 2.636 6.364 2.636c2.125 0 4.078-0.737 5.618-1.968l3.675 3.675c0.391 0.391 1.024 0.391 1.414 0s0.391-1.024 0-1.414z');
    element.appendChild(path);
    return element;
};

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"71EHb":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "createEffectWrapper", ()=>createEffectWrapper
);
function createEffectWrapper() {
    var effects = [];
    var cleanups = [];
    function runEffect(fn) {
        effects.push(fn);
        var effectCleanup = fn();
        cleanups.push(effectCleanup);
    }
    return {
        runEffect: runEffect,
        cleanupEffects: function cleanupEffects() {
            var currentCleanups = cleanups;
            cleanups = [];
            currentCleanups.forEach(function(cleanup) {
                cleanup();
            });
        },
        runEffects: function runEffects() {
            var currentEffects = effects;
            effects = [];
            currentEffects.forEach(function(effect) {
                runEffect(effect);
            });
        }
    };
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"grTGt":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "createReactiveWrapper", ()=>createReactiveWrapper
);
function createReactiveWrapper() {
    var reactives = [];
    return {
        reactive: function reactive(value1) {
            var current = value1();
            var reactive = {
                _fn: value1,
                _ref: {
                    current: current
                },
                get value () {
                    return this._ref.current;
                },
                set value (value){
                    this._ref.current = value;
                }
            };
            reactives.push(reactive);
            return reactive;
        },
        runReactives: function runReactives() {
            reactives.forEach(function(value) {
                value._ref.current = value._fn();
            });
        }
    };
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"ei4ti":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "getDefaultOptions", ()=>getDefaultOptions
);
var _autocompleteShared = require("@algolia/autocomplete-shared");
var _preact = require("preact");
var _components = require("./components");
var _utils = require("./utils");
var _excluded = [
    "classNames",
    "container",
    "getEnvironmentProps",
    "getFormProps",
    "getInputProps",
    "getItemProps",
    "getLabelProps",
    "getListProps",
    "getPanelProps",
    "getRootProps",
    "panelContainer",
    "panelPlacement",
    "render",
    "renderNoResults",
    "renderer",
    "detachedMediaQuery",
    "components",
    "translations"
];
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        enumerableOnly && (symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        })), keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = null != arguments[i] ? arguments[i] : {};
        i % 2 ? ownKeys(Object(source), !0).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
function _objectWithoutProperties(source, excluded) {
    if (source == null) return {};
    var target = _objectWithoutPropertiesLoose(source, excluded);
    var key, i;
    if (Object.getOwnPropertySymbols) {
        var sourceSymbolKeys = Object.getOwnPropertySymbols(source);
        for(i = 0; i < sourceSymbolKeys.length; i++){
            key = sourceSymbolKeys[i];
            if (excluded.indexOf(key) >= 0) continue;
            if (!Object.prototype.propertyIsEnumerable.call(source, key)) continue;
            target[key] = source[key];
        }
    }
    return target;
}
function _objectWithoutPropertiesLoose(source, excluded) {
    if (source == null) return {};
    var target = {};
    var sourceKeys = Object.keys(source);
    var key, i;
    for(i = 0; i < sourceKeys.length; i++){
        key = sourceKeys[i];
        if (excluded.indexOf(key) >= 0) continue;
        target[key] = source[key];
    }
    return target;
}
var defaultClassNames = {
    clearButton: 'aa-ClearButton',
    detachedCancelButton: 'aa-DetachedCancelButton',
    detachedContainer: 'aa-DetachedContainer',
    detachedFormContainer: 'aa-DetachedFormContainer',
    detachedOverlay: 'aa-DetachedOverlay',
    detachedSearchButton: 'aa-DetachedSearchButton',
    detachedSearchButtonIcon: 'aa-DetachedSearchButtonIcon',
    detachedSearchButtonPlaceholder: 'aa-DetachedSearchButtonPlaceholder',
    form: 'aa-Form',
    input: 'aa-Input',
    inputWrapper: 'aa-InputWrapper',
    inputWrapperPrefix: 'aa-InputWrapperPrefix',
    inputWrapperSuffix: 'aa-InputWrapperSuffix',
    item: 'aa-Item',
    label: 'aa-Label',
    list: 'aa-List',
    loadingIndicator: 'aa-LoadingIndicator',
    panel: 'aa-Panel',
    panelLayout: 'aa-PanelLayout aa-Panel--scrollable',
    root: 'aa-Autocomplete',
    source: 'aa-Source',
    sourceFooter: 'aa-SourceFooter',
    sourceHeader: 'aa-SourceHeader',
    sourceNoResults: 'aa-SourceNoResults',
    submitButton: 'aa-SubmitButton'
};
var defaultRender = function defaultRender(_ref, root) {
    var children = _ref.children, render = _ref.render;
    render(children, root);
};
var defaultRenderer = {
    createElement: _preact.createElement,
    Fragment: _preact.Fragment,
    render: _preact.render
};
function getDefaultOptions(options) {
    var _core$id;
    var classNames = options.classNames, container = options.container, getEnvironmentProps = options.getEnvironmentProps, getFormProps = options.getFormProps, getInputProps = options.getInputProps, getItemProps = options.getItemProps, getLabelProps = options.getLabelProps, getListProps = options.getListProps, getPanelProps = options.getPanelProps, getRootProps = options.getRootProps, panelContainer = options.panelContainer, panelPlacement = options.panelPlacement, render = options.render, renderNoResults = options.renderNoResults, renderer = options.renderer, detachedMediaQuery = options.detachedMediaQuery, components = options.components, translations = options.translations, core = _objectWithoutProperties(options, _excluded);
    /* eslint-disable no-restricted-globals */ var environment = typeof window !== 'undefined' ? window : {};
    /* eslint-enable no-restricted-globals */ var containerElement = _utils.getHTMLElement(environment, container);
    _autocompleteShared.invariant(containerElement.tagName !== 'INPUT', 'The `container` option does not support `input` elements. You need to change the container to a `div`.');
    _autocompleteShared.warn(!(render && renderer && !(renderer !== null && renderer !== void 0 && renderer.render)), "You provided the `render` option but did not provide a `renderer.render`. Since v1.6.0, you can provide a `render` function directly in `renderer`.\nTo get rid of this warning, do any of the following depending on your use case.\n- If you are using the `render` option only to override Autocomplete's default `render` function, pass the `render` function into `renderer` and remove the `render` option.\n- If you are using the `render` option to customize the layout, pass your `render` function into `renderer` and use it from the provided parameters of the `render` option.\n- If you are using the `render` option to work with React 18, pass an empty `render` function into `renderer`.\nSee https://www.algolia.com/doc/ui-libraries/autocomplete/api-reference/autocomplete-js/autocomplete/#param-render");
    _autocompleteShared.warn(!renderer || render || renderer.Fragment && renderer.createElement && renderer.render, "You provided an incomplete `renderer` (missing: ".concat([
        !(renderer !== null && renderer !== void 0 && renderer.createElement) && '`renderer.createElement`',
        !(renderer !== null && renderer !== void 0 && renderer.Fragment) && '`renderer.Fragment`',
        !(renderer !== null && renderer !== void 0 && renderer.render) && '`renderer.render`'
    ].filter(Boolean).join(', '), "). This can cause rendering issues.") + '\nSee https://www.algolia.com/doc/ui-libraries/autocomplete/api-reference/autocomplete-js/autocomplete/#param-renderer');
    var defaultedRenderer = _objectSpread(_objectSpread({}, defaultRenderer), renderer);
    var defaultComponents = {
        Highlight: _components.createHighlightComponent(defaultedRenderer),
        ReverseHighlight: _components.createReverseHighlightComponent(defaultedRenderer),
        ReverseSnippet: _components.createReverseSnippetComponent(defaultedRenderer),
        Snippet: _components.createSnippetComponent(defaultedRenderer)
    };
    var defaultTranslations = {
        clearButtonTitle: 'Clear',
        detachedCancelButtonText: 'Cancel',
        submitButtonTitle: 'Submit'
    };
    return {
        renderer: {
            classNames: _utils.mergeClassNames(defaultClassNames, classNames !== null && classNames !== void 0 ? classNames : {}),
            container: containerElement,
            getEnvironmentProps: getEnvironmentProps !== null && getEnvironmentProps !== void 0 ? getEnvironmentProps : function(_ref2) {
                var props = _ref2.props;
                return props;
            },
            getFormProps: getFormProps !== null && getFormProps !== void 0 ? getFormProps : function(_ref3) {
                var props = _ref3.props;
                return props;
            },
            getInputProps: getInputProps !== null && getInputProps !== void 0 ? getInputProps : function(_ref4) {
                var props = _ref4.props;
                return props;
            },
            getItemProps: getItemProps !== null && getItemProps !== void 0 ? getItemProps : function(_ref5) {
                var props = _ref5.props;
                return props;
            },
            getLabelProps: getLabelProps !== null && getLabelProps !== void 0 ? getLabelProps : function(_ref6) {
                var props = _ref6.props;
                return props;
            },
            getListProps: getListProps !== null && getListProps !== void 0 ? getListProps : function(_ref7) {
                var props = _ref7.props;
                return props;
            },
            getPanelProps: getPanelProps !== null && getPanelProps !== void 0 ? getPanelProps : function(_ref8) {
                var props = _ref8.props;
                return props;
            },
            getRootProps: getRootProps !== null && getRootProps !== void 0 ? getRootProps : function(_ref9) {
                var props = _ref9.props;
                return props;
            },
            panelContainer: panelContainer ? _utils.getHTMLElement(environment, panelContainer) : environment.document.body,
            panelPlacement: panelPlacement !== null && panelPlacement !== void 0 ? panelPlacement : 'input-wrapper-width',
            render: render !== null && render !== void 0 ? render : defaultRender,
            renderNoResults: renderNoResults,
            renderer: defaultedRenderer,
            detachedMediaQuery: detachedMediaQuery !== null && detachedMediaQuery !== void 0 ? detachedMediaQuery : getComputedStyle(environment.document.documentElement).getPropertyValue('--aa-detached-media-query'),
            components: _objectSpread(_objectSpread({}, defaultComponents), components),
            translations: _objectSpread(_objectSpread({}, defaultTranslations), translations)
        },
        core: _objectSpread(_objectSpread({}, core), {}, {
            id: (_core$id = core.id) !== null && _core$id !== void 0 ? _core$id : _autocompleteShared.generateAutocompleteId(),
            environment: environment
        })
    };
}

},{"@algolia/autocomplete-shared":"59T59","preact":"26zcy","./components":"5K3sr","./utils":"fKU1x","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"26zcy":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "render", ()=>N
);
parcelHelpers.export(exports, "hydrate", ()=>O
);
parcelHelpers.export(exports, "createElement", ()=>a
);
parcelHelpers.export(exports, "h", ()=>a
);
parcelHelpers.export(exports, "Fragment", ()=>y
);
parcelHelpers.export(exports, "createRef", ()=>h
);
parcelHelpers.export(exports, "isValidElement", ()=>l
);
parcelHelpers.export(exports, "Component", ()=>p
);
parcelHelpers.export(exports, "cloneElement", ()=>S
);
parcelHelpers.export(exports, "createContext", ()=>q
);
parcelHelpers.export(exports, "toChildArray", ()=>w
);
parcelHelpers.export(exports, "options", ()=>n
);
var n, l, u, i, t, o, r = {}, f = [], e = /acit|ex(?:s|g|n|p|$)|rph|grid|ows|mnc|ntw|ine[ch]|zoo|^ord|itera/i;
function c(n1, l1) {
    for(var u1 in l1)n1[u1] = l1[u1];
    return n1;
}
function s(n2) {
    var l2 = n2.parentNode;
    l2 && l2.removeChild(n2);
}
function a(n3, l3, u2) {
    var i1, t1, o1, r1 = arguments, f1 = {};
    for(o1 in l3)"key" == o1 ? i1 = l3[o1] : "ref" == o1 ? t1 = l3[o1] : f1[o1] = l3[o1];
    if (arguments.length > 3) for(u2 = [
        u2
    ], o1 = 3; o1 < arguments.length; o1++)u2.push(r1[o1]);
    if (null != u2 && (f1.children = u2), "function" == typeof n3 && null != n3.defaultProps) for(o1 in n3.defaultProps)void 0 === f1[o1] && (f1[o1] = n3.defaultProps[o1]);
    return v(n3, f1, i1, t1, null);
}
function v(l4, u3, i2, t2, o2) {
    var r2 = {
        type: l4,
        props: u3,
        key: i2,
        ref: t2,
        __k: null,
        __: null,
        __b: 0,
        __e: null,
        __d: void 0,
        __c: null,
        __h: null,
        constructor: void 0,
        __v: null == o2 ? ++n.__v : o2
    };
    return null != n.vnode && n.vnode(r2), r2;
}
function h() {
    return {
        current: null
    };
}
function y(n4) {
    return n4.children;
}
function p(n5, l5) {
    this.props = n5, this.context = l5;
}
function d(n6, l6) {
    if (null == l6) return n6.__ ? d(n6.__, n6.__.__k.indexOf(n6) + 1) : null;
    for(var u4; l6 < n6.__k.length; l6++)if (null != (u4 = n6.__k[l6]) && null != u4.__e) return u4.__e;
    return "function" == typeof n6.type ? d(n6) : null;
}
function _(n7) {
    var l7, u5;
    if (null != (n7 = n7.__) && null != n7.__c) {
        for(n7.__e = n7.__c.base = null, l7 = 0; l7 < n7.__k.length; l7++)if (null != (u5 = n7.__k[l7]) && null != u5.__e) {
            n7.__e = n7.__c.base = u5.__e;
            break;
        }
        return _(n7);
    }
}
function k(l8) {
    (!l8.__d && (l8.__d = !0) && u.push(l8) && !b.__r++ || t !== n.debounceRendering) && ((t = n.debounceRendering) || i)(b);
}
function b() {
    for(var n8; b.__r = u.length;)n8 = u.sort(function(n9, l9) {
        return n9.__v.__b - l9.__v.__b;
    }), u = [], n8.some(function(n10) {
        var l10, u6, i3, t3, o3, r3;
        n10.__d && (o3 = (t3 = (l10 = n10).__v).__e, (r3 = l10.__P) && (u6 = [], (i3 = c({}, t3)).__v = t3.__v + 1, I(r3, t3, i3, l10.__n, void 0 !== r3.ownerSVGElement, null != t3.__h ? [
            o3
        ] : null, u6, null == o3 ? d(t3) : o3, t3.__h), T(u6, t3), t3.__e != o3 && _(t3)));
    });
}
function m(n11, l11, u7, i4, t4, o4, e1, c1, s1, a1) {
    var h1, p1, _1, k1, b1, m1, w1, A1 = i4 && i4.__k || f, P1 = A1.length;
    for(u7.__k = [], h1 = 0; h1 < l11.length; h1++)if (null != (k1 = u7.__k[h1] = null == (k1 = l11[h1]) || "boolean" == typeof k1 ? null : "string" == typeof k1 || "number" == typeof k1 || "bigint" == typeof k1 ? v(null, k1, null, null, k1) : Array.isArray(k1) ? v(y, {
        children: k1
    }, null, null, null) : k1.__b > 0 ? v(k1.type, k1.props, k1.key, null, k1.__v) : k1)) {
        if (k1.__ = u7, k1.__b = u7.__b + 1, null === (_1 = A1[h1]) || _1 && k1.key == _1.key && k1.type === _1.type) A1[h1] = void 0;
        else for(p1 = 0; p1 < P1; p1++){
            if ((_1 = A1[p1]) && k1.key == _1.key && k1.type === _1.type) {
                A1[p1] = void 0;
                break;
            }
            _1 = null;
        }
        I(n11, k1, _1 = _1 || r, t4, o4, e1, c1, s1, a1), b1 = k1.__e, (p1 = k1.ref) && _1.ref != p1 && (w1 || (w1 = []), _1.ref && w1.push(_1.ref, null, k1), w1.push(p1, k1.__c || b1, k1)), null != b1 ? (null == m1 && (m1 = b1), "function" == typeof k1.type && null != k1.__k && k1.__k === _1.__k ? k1.__d = s1 = g(k1, s1, n11) : s1 = x(n11, k1, _1, A1, b1, s1), a1 || "option" !== u7.type ? "function" == typeof u7.type && (u7.__d = s1) : n11.value = "") : s1 && _1.__e == s1 && s1.parentNode != n11 && (s1 = d(_1));
    }
    for(u7.__e = m1, h1 = P1; h1--;)null != A1[h1] && ("function" == typeof u7.type && null != A1[h1].__e && A1[h1].__e == u7.__d && (u7.__d = d(i4, h1 + 1)), L(A1[h1], A1[h1]));
    if (w1) for(h1 = 0; h1 < w1.length; h1++)z(w1[h1], w1[++h1], w1[++h1]);
}
function g(n12, l12, u8) {
    var i5, t5;
    for(i5 = 0; i5 < n12.__k.length; i5++)(t5 = n12.__k[i5]) && (t5.__ = n12, l12 = "function" == typeof t5.type ? g(t5, l12, u8) : x(u8, t5, t5, n12.__k, t5.__e, l12));
    return l12;
}
function w(n13, l13) {
    return l13 = l13 || [], null == n13 || "boolean" == typeof n13 || (Array.isArray(n13) ? n13.some(function(n14) {
        w(n14, l13);
    }) : l13.push(n13)), l13;
}
function x(n15, l14, u9, i6, t6, o5) {
    var r4, f2, e2;
    if (void 0 !== l14.__d) r4 = l14.__d, l14.__d = void 0;
    else if (null == u9 || t6 != o5 || null == t6.parentNode) n: if (null == o5 || o5.parentNode !== n15) n15.appendChild(t6), r4 = null;
    else {
        for(f2 = o5, e2 = 0; (f2 = f2.nextSibling) && e2 < i6.length; e2 += 2)if (f2 == t6) break n;
        n15.insertBefore(t6, o5), r4 = o5;
    }
    return void 0 !== r4 ? r4 : t6.nextSibling;
}
function A(n16, l15, u10, i7, t7) {
    var o6;
    for(o6 in u10)"children" === o6 || "key" === o6 || o6 in l15 || C(n16, o6, null, u10[o6], i7);
    for(o6 in l15)t7 && "function" != typeof l15[o6] || "children" === o6 || "key" === o6 || "value" === o6 || "checked" === o6 || u10[o6] === l15[o6] || C(n16, o6, l15[o6], u10[o6], i7);
}
function P(n17, l16, u11) {
    "-" === l16[0] ? n17.setProperty(l16, u11) : n17[l16] = null == u11 ? "" : "number" != typeof u11 || e.test(l16) ? u11 : u11 + "px";
}
function C(n18, l17, u12, i8, t8) {
    var o7;
    n: if ("style" === l17) {
        if ("string" == typeof u12) n18.style.cssText = u12;
        else {
            if ("string" == typeof i8 && (n18.style.cssText = i8 = ""), i8) for(l17 in i8)u12 && l17 in u12 || P(n18.style, l17, "");
            if (u12) for(l17 in u12)i8 && u12[l17] === i8[l17] || P(n18.style, l17, u12[l17]);
        }
    } else if ("o" === l17[0] && "n" === l17[1]) o7 = l17 !== (l17 = l17.replace(/Capture$/, "")), l17 = l17.toLowerCase() in n18 ? l17.toLowerCase().slice(2) : l17.slice(2), n18.l || (n18.l = {}), n18.l[l17 + o7] = u12, u12 ? i8 || n18.addEventListener(l17, o7 ? H : $, o7) : n18.removeEventListener(l17, o7 ? H : $, o7);
    else if ("dangerouslySetInnerHTML" !== l17) {
        if (t8) l17 = l17.replace(/xlink[H:h]/, "h").replace(/sName$/, "s");
        else if ("href" !== l17 && "list" !== l17 && "form" !== l17 && "tabIndex" !== l17 && "download" !== l17 && l17 in n18) try {
            n18[l17] = null == u12 ? "" : u12;
            break n;
        } catch (n) {}
        "function" == typeof u12 || (null != u12 && (!1 !== u12 || "a" === l17[0] && "r" === l17[1]) ? n18.setAttribute(l17, u12) : n18.removeAttribute(l17));
    }
}
function $(l18) {
    this.l[l18.type + !1](n.event ? n.event(l18) : l18);
}
function H(l19) {
    this.l[l19.type + !0](n.event ? n.event(l19) : l19);
}
function I(l20, u13, i9, t9, o8, r5, f3, e3, s2) {
    var a2, v1, h2, d1, _2, k2, b2, g1, w2, x1, A2, P2 = u13.type;
    if (void 0 !== u13.constructor) return null;
    null != i9.__h && (s2 = i9.__h, e3 = u13.__e = i9.__e, u13.__h = null, r5 = [
        e3
    ]), (a2 = n.__b) && a2(u13);
    try {
        n: if ("function" == typeof P2) {
            if (g1 = u13.props, w2 = (a2 = P2.contextType) && t9[a2.__c], x1 = a2 ? w2 ? w2.props.value : a2.__ : t9, i9.__c ? b2 = (v1 = u13.__c = i9.__c).__ = v1.__E : ("prototype" in P2 && P2.prototype.render ? u13.__c = v1 = new P2(g1, x1) : (u13.__c = v1 = new p(g1, x1), v1.constructor = P2, v1.render = M), w2 && w2.sub(v1), v1.props = g1, v1.state || (v1.state = {}), v1.context = x1, v1.__n = t9, h2 = v1.__d = !0, v1.__h = []), null == v1.__s && (v1.__s = v1.state), null != P2.getDerivedStateFromProps && (v1.__s == v1.state && (v1.__s = c({}, v1.__s)), c(v1.__s, P2.getDerivedStateFromProps(g1, v1.__s))), d1 = v1.props, _2 = v1.state, h2) null == P2.getDerivedStateFromProps && null != v1.componentWillMount && v1.componentWillMount(), null != v1.componentDidMount && v1.__h.push(v1.componentDidMount);
            else {
                if (null == P2.getDerivedStateFromProps && g1 !== d1 && null != v1.componentWillReceiveProps && v1.componentWillReceiveProps(g1, x1), !v1.__e && null != v1.shouldComponentUpdate && !1 === v1.shouldComponentUpdate(g1, v1.__s, x1) || u13.__v === i9.__v) {
                    v1.props = g1, v1.state = v1.__s, u13.__v !== i9.__v && (v1.__d = !1), v1.__v = u13, u13.__e = i9.__e, u13.__k = i9.__k, u13.__k.forEach(function(n19) {
                        n19 && (n19.__ = u13);
                    }), v1.__h.length && f3.push(v1);
                    break n;
                }
                null != v1.componentWillUpdate && v1.componentWillUpdate(g1, v1.__s, x1), null != v1.componentDidUpdate && v1.__h.push(function() {
                    v1.componentDidUpdate(d1, _2, k2);
                });
            }
            v1.context = x1, v1.props = g1, v1.state = v1.__s, (a2 = n.__r) && a2(u13), v1.__d = !1, v1.__v = u13, v1.__P = l20, a2 = v1.render(v1.props, v1.state, v1.context), v1.state = v1.__s, null != v1.getChildContext && (t9 = c(c({}, t9), v1.getChildContext())), h2 || null == v1.getSnapshotBeforeUpdate || (k2 = v1.getSnapshotBeforeUpdate(d1, _2)), A2 = null != a2 && a2.type === y && null == a2.key ? a2.props.children : a2, m(l20, Array.isArray(A2) ? A2 : [
                A2
            ], u13, i9, t9, o8, r5, f3, e3, s2), v1.base = u13.__e, u13.__h = null, v1.__h.length && f3.push(v1), b2 && (v1.__E = v1.__ = null), v1.__e = !1;
        } else null == r5 && u13.__v === i9.__v ? (u13.__k = i9.__k, u13.__e = i9.__e) : u13.__e = j(i9.__e, u13, i9, t9, o8, r5, f3, s2);
        (a2 = n.diffed) && a2(u13);
    } catch (l21) {
        u13.__v = null, (s2 || null != r5) && (u13.__e = e3, u13.__h = !!s2, r5[r5.indexOf(e3)] = null), n.__e(l21, u13, i9);
    }
}
function T(l22, u14) {
    n.__c && n.__c(u14, l22), l22.some(function(u15) {
        try {
            l22 = u15.__h, u15.__h = [], l22.some(function(n20) {
                n20.call(u15);
            });
        } catch (l23) {
            n.__e(l23, u15.__v);
        }
    });
}
function j(n21, l24, u16, i10, t10, o9, e4, c2) {
    var a3, v2, h3, y1, p2 = u16.props, d2 = l24.props, _3 = l24.type, k3 = 0;
    if ("svg" === _3 && (t10 = !0), null != o9) {
        for(; k3 < o9.length; k3++)if ((a3 = o9[k3]) && (a3 === n21 || (_3 ? a3.localName == _3 : 3 == a3.nodeType))) {
            n21 = a3, o9[k3] = null;
            break;
        }
    }
    if (null == n21) {
        if (null === _3) return document.createTextNode(d2);
        n21 = t10 ? document.createElementNS("http://www.w3.org/2000/svg", _3) : document.createElement(_3, d2.is && d2), o9 = null, c2 = !1;
    }
    if (null === _3) p2 === d2 || c2 && n21.data === d2 || (n21.data = d2);
    else {
        if (o9 = o9 && f.slice.call(n21.childNodes), v2 = (p2 = u16.props || r).dangerouslySetInnerHTML, h3 = d2.dangerouslySetInnerHTML, !c2) {
            if (null != o9) for(p2 = {}, y1 = 0; y1 < n21.attributes.length; y1++)p2[n21.attributes[y1].name] = n21.attributes[y1].value;
            (h3 || v2) && (h3 && (v2 && h3.__html == v2.__html || h3.__html === n21.innerHTML) || (n21.innerHTML = h3 && h3.__html || ""));
        }
        if (A(n21, d2, p2, t10, c2), h3) l24.__k = [];
        else if (k3 = l24.props.children, m(n21, Array.isArray(k3) ? k3 : [
            k3
        ], l24, u16, i10, t10 && "foreignObject" !== _3, o9, e4, n21.firstChild, c2), null != o9) for(k3 = o9.length; k3--;)null != o9[k3] && s(o9[k3]);
        c2 || ("value" in d2 && void 0 !== (k3 = d2.value) && (k3 !== n21.value || "progress" === _3 && !k3) && C(n21, "value", k3, p2.value, !1), "checked" in d2 && void 0 !== (k3 = d2.checked) && k3 !== n21.checked && C(n21, "checked", k3, p2.checked, !1));
    }
    return n21;
}
function z(l25, u17, i11) {
    try {
        "function" == typeof l25 ? l25(u17) : l25.current = u17;
    } catch (l26) {
        n.__e(l26, i11);
    }
}
function L(l27, u18, i12) {
    var t11, o10, r6;
    if (n.unmount && n.unmount(l27), (t11 = l27.ref) && (t11.current && t11.current !== l27.__e || z(t11, null, u18)), i12 || "function" == typeof l27.type || (i12 = null != (o10 = l27.__e)), l27.__e = l27.__d = void 0, null != (t11 = l27.__c)) {
        if (t11.componentWillUnmount) try {
            t11.componentWillUnmount();
        } catch (l28) {
            n.__e(l28, u18);
        }
        t11.base = t11.__P = null;
    }
    if (t11 = l27.__k) for(r6 = 0; r6 < t11.length; r6++)t11[r6] && L(t11[r6], u18, i12);
    null != o10 && s(o10);
}
function M(n22, l, u19) {
    return this.constructor(n22, u19);
}
function N(l29, u20, i13) {
    var t12, o11, e5;
    n.__ && n.__(l29, u20), o11 = (t12 = "function" == typeof i13) ? null : i13 && i13.__k || u20.__k, e5 = [], I(u20, l29 = (!t12 && i13 || u20).__k = a(y, null, [
        l29
    ]), o11 || r, r, void 0 !== u20.ownerSVGElement, !t12 && i13 ? [
        i13
    ] : o11 ? null : u20.firstChild ? f.slice.call(u20.childNodes) : null, e5, !t12 && i13 ? i13 : o11 ? o11.__e : u20.firstChild, t12), T(e5, l29);
}
function O(n23, l30) {
    N(n23, l30, O);
}
function S(n24, l31, u21) {
    var i14, t13, o12, r7 = arguments, f4 = c({}, n24.props);
    for(o12 in l31)"key" == o12 ? i14 = l31[o12] : "ref" == o12 ? t13 = l31[o12] : f4[o12] = l31[o12];
    if (arguments.length > 3) for(u21 = [
        u21
    ], o12 = 3; o12 < arguments.length; o12++)u21.push(r7[o12]);
    return null != u21 && (f4.children = u21), v(n24.type, f4, i14 || n24.key, t13 || n24.ref, null);
}
function q(n25, l32) {
    var u22 = {
        __c: l32 = "__cC" + o++,
        __: n25,
        Consumer: function(n26, l33) {
            return n26.children(l33);
        },
        Provider: function(n27) {
            var u23, i15;
            return this.getChildContext || (u23 = [], (i15 = {})[l32] = this, this.getChildContext = function() {
                return i15;
            }, this.shouldComponentUpdate = function(n28) {
                this.props.value !== n28.value && u23.some(k);
            }, this.sub = function(n29) {
                u23.push(n29);
                var l34 = n29.componentWillUnmount;
                n29.componentWillUnmount = function() {
                    u23.splice(u23.indexOf(n29), 1), l34 && l34.call(n29);
                };
            }), n27.children;
        }
    };
    return u22.Provider.__ = u22.Consumer.contextType = u22;
}
n = {
    __e: function(n30, l35) {
        for(var u24, i16, t14; l35 = l35.__;)if ((u24 = l35.__c) && !u24.__) try {
            if ((i16 = u24.constructor) && null != i16.getDerivedStateFromError && (u24.setState(i16.getDerivedStateFromError(n30)), t14 = u24.__d), null != u24.componentDidCatch && (u24.componentDidCatch(n30), t14 = u24.__d), t14) return u24.__E = u24;
        } catch (l36) {
            n30 = l36;
        }
        throw n30;
    },
    __v: 0
}, l = function(n31) {
    return null != n31 && void 0 === n31.constructor;
}, p.prototype.setState = function(n32, l37) {
    var u25;
    u25 = null != this.__s && this.__s !== this.state ? this.__s : this.__s = c({}, this.state), "function" == typeof n32 && (n32 = n32(c({}, u25), this.props)), n32 && c(u25, n32), null != n32 && this.__v && (l37 && this.__h.push(l37), k(this));
}, p.prototype.forceUpdate = function(n33) {
    this.__v && (this.__e = !0, n33 && this.__h.push(n33), k(this));
}, p.prototype.render = y, u = [], i = "function" == typeof Promise ? Promise.prototype.then.bind(Promise.resolve()) : setTimeout, b.__r = 0, o = 0;

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"5K3sr":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _highlight = require("./Highlight");
parcelHelpers.exportAll(_highlight, exports);
var _reverseHighlight = require("./ReverseHighlight");
parcelHelpers.exportAll(_reverseHighlight, exports);
var _reverseSnippet = require("./ReverseSnippet");
parcelHelpers.exportAll(_reverseSnippet, exports);
var _snippet = require("./Snippet");
parcelHelpers.exportAll(_snippet, exports);

},{"./Highlight":"bKGip","./ReverseHighlight":"b85bW","./ReverseSnippet":"fNt5i","./Snippet":"bThlx","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"bKGip":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "createHighlightComponent", ()=>createHighlightComponent
);
var _autocompletePresetAlgolia = require("@algolia/autocomplete-preset-algolia");
function createHighlightComponent(_ref) {
    var createElement = _ref.createElement, Fragment = _ref.Fragment;
    function Highlight(_ref2) {
        var hit = _ref2.hit, attribute = _ref2.attribute, _ref2$tagName = _ref2.tagName, tagName = _ref2$tagName === void 0 ? 'mark' : _ref2$tagName;
        return createElement(Fragment, {}, _autocompletePresetAlgolia.parseAlgoliaHitHighlight({
            hit: hit,
            attribute: attribute
        }).map(function(x, index) {
            return x.isHighlighted ? createElement(tagName, {
                key: index
            }, x.value) : x.value;
        }));
    }
    Highlight.__autocomplete_componentName = 'Highlight';
    return Highlight;
}

},{"@algolia/autocomplete-preset-algolia":"lp3lw","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"lp3lw":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _highlight = require("./highlight");
parcelHelpers.exportAll(_highlight, exports);
var _requester = require("./requester");
parcelHelpers.exportAll(_requester, exports);
var _search = require("./search");
parcelHelpers.exportAll(_search, exports);

},{"./highlight":"5Fy50","./requester":"a8CYC","./search":"gwNho","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"5Fy50":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _highlightedHit = require("./HighlightedHit");
parcelHelpers.exportAll(_highlightedHit, exports);
var _parseAlgoliaHitHighlight = require("./parseAlgoliaHitHighlight");
parcelHelpers.exportAll(_parseAlgoliaHitHighlight, exports);
var _parseAlgoliaHitReverseHighlight = require("./parseAlgoliaHitReverseHighlight");
parcelHelpers.exportAll(_parseAlgoliaHitReverseHighlight, exports);
var _parseAlgoliaHitReverseSnippet = require("./parseAlgoliaHitReverseSnippet");
parcelHelpers.exportAll(_parseAlgoliaHitReverseSnippet, exports);
var _parseAlgoliaHitSnippet = require("./parseAlgoliaHitSnippet");
parcelHelpers.exportAll(_parseAlgoliaHitSnippet, exports);
var _snippetedHit = require("./SnippetedHit");
parcelHelpers.exportAll(_snippetedHit, exports);

},{"./HighlightedHit":"esH8u","./parseAlgoliaHitHighlight":"8oRZg","./parseAlgoliaHitReverseHighlight":"10MVc","./parseAlgoliaHitReverseSnippet":"hDmRU","./parseAlgoliaHitSnippet":"e5gtH","./SnippetedHit":"cZ7Rk","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"esH8u":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"8oRZg":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "parseAlgoliaHitHighlight", ()=>parseAlgoliaHitHighlight
);
var _autocompleteShared = require("@algolia/autocomplete-shared");
var _parseAttribute = require("./parseAttribute");
function _toConsumableArray(arr) {
    return _arrayWithoutHoles(arr) || _iterableToArray(arr) || _unsupportedIterableToArray(arr) || _nonIterableSpread();
}
function _nonIterableSpread() {
    throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
}
function _unsupportedIterableToArray(o, minLen) {
    if (!o) return;
    if (typeof o === "string") return _arrayLikeToArray(o, minLen);
    var n = Object.prototype.toString.call(o).slice(8, -1);
    if (n === "Object" && o.constructor) n = o.constructor.name;
    if (n === "Map" || n === "Set") return Array.from(o);
    if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen);
}
function _iterableToArray(iter) {
    if (typeof Symbol !== "undefined" && iter[Symbol.iterator] != null || iter["@@iterator"] != null) return Array.from(iter);
}
function _arrayWithoutHoles(arr) {
    if (Array.isArray(arr)) return _arrayLikeToArray(arr);
}
function _arrayLikeToArray(arr, len) {
    if (len == null || len > arr.length) len = arr.length;
    for(var i = 0, arr2 = new Array(len); i < len; i++)arr2[i] = arr[i];
    return arr2;
}
function parseAlgoliaHitHighlight(_ref) {
    var hit = _ref.hit, attribute = _ref.attribute;
    var path = Array.isArray(attribute) ? attribute : [
        attribute
    ];
    var highlightedValue = _autocompleteShared.getAttributeValueByPath(hit, [
        '_highlightResult'
    ].concat(_toConsumableArray(path), [
        'value'
    ]));
    if (typeof highlightedValue !== 'string') {
        _autocompleteShared.warn(false, "The attribute \"".concat(path.join('.'), "\" described by the path ").concat(JSON.stringify(path), " does not exist on the hit. Did you set it in `attributesToHighlight`?") + '\nSee https://www.algolia.com/doc/api-reference/api-parameters/attributesToHighlight/');
        highlightedValue = _autocompleteShared.getAttributeValueByPath(hit, path) || '';
    }
    return _parseAttribute.parseAttribute({
        highlightedValue: highlightedValue
    });
}

},{"@algolia/autocomplete-shared":"59T59","./parseAttribute":"3rO3A","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"3rO3A":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "parseAttribute", ()=>parseAttribute
);
var _constants = require("../constants");
/**
 * Creates a data structure that allows to concatenate similar highlighting
 * parts in a single value.
 */ function createAttributeSet() {
    var initialValue = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : [];
    var value = initialValue;
    return {
        get: function get() {
            return value;
        },
        add: function add(part) {
            var lastPart = value[value.length - 1];
            if ((lastPart === null || lastPart === void 0 ? void 0 : lastPart.isHighlighted) === part.isHighlighted) value[value.length - 1] = {
                value: lastPart.value + part.value,
                isHighlighted: lastPart.isHighlighted
            };
            else value.push(part);
        }
    };
}
function parseAttribute(_ref) {
    var highlightedValue = _ref.highlightedValue;
    var preTagParts = highlightedValue.split(_constants.HIGHLIGHT_PRE_TAG);
    var firstValue = preTagParts.shift();
    var parts = createAttributeSet(firstValue ? [
        {
            value: firstValue,
            isHighlighted: false
        }
    ] : []);
    preTagParts.forEach(function(part) {
        var postTagParts = part.split(_constants.HIGHLIGHT_POST_TAG);
        parts.add({
            value: postTagParts[0],
            isHighlighted: true
        });
        if (postTagParts[1] !== '') parts.add({
            value: postTagParts[1],
            isHighlighted: false
        });
    });
    return parts.get();
}

},{"../constants":"hThxU","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"hThxU":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "HIGHLIGHT_PRE_TAG", ()=>HIGHLIGHT_PRE_TAG
);
parcelHelpers.export(exports, "HIGHLIGHT_POST_TAG", ()=>HIGHLIGHT_POST_TAG
);
var HIGHLIGHT_PRE_TAG = '__aa-highlight__';
var HIGHLIGHT_POST_TAG = '__/aa-highlight__';

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"10MVc":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "parseAlgoliaHitReverseHighlight", ()=>parseAlgoliaHitReverseHighlight
);
var _parseAlgoliaHitHighlight = require("./parseAlgoliaHitHighlight");
var _reverseHighlightedParts = require("./reverseHighlightedParts");
function parseAlgoliaHitReverseHighlight(props) {
    return _reverseHighlightedParts.reverseHighlightedParts(_parseAlgoliaHitHighlight.parseAlgoliaHitHighlight(props));
}

},{"./parseAlgoliaHitHighlight":"8oRZg","./reverseHighlightedParts":"lnI4J","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"lnI4J":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "reverseHighlightedParts", ()=>reverseHighlightedParts
);
var _isPartHighlighted = require("./isPartHighlighted");
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        enumerableOnly && (symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        })), keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = null != arguments[i] ? arguments[i] : {};
        i % 2 ? ownKeys(Object(source), !0).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
function reverseHighlightedParts(parts) {
    // We don't want to highlight the whole word when no parts match.
    if (!parts.some(function(part) {
        return part.isHighlighted;
    })) return parts.map(function(part) {
        return _objectSpread(_objectSpread({}, part), {}, {
            isHighlighted: false
        });
    });
    return parts.map(function(part, i) {
        return _objectSpread(_objectSpread({}, part), {}, {
            isHighlighted: !_isPartHighlighted.isPartHighlighted(parts, i)
        });
    });
}

},{"./isPartHighlighted":"hFlzN","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"hFlzN":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "isPartHighlighted", ()=>isPartHighlighted
);
var htmlEscapes = {
    '&amp;': '&',
    '&lt;': '<',
    '&gt;': '>',
    '&quot;': '"',
    '&#39;': "'"
};
var hasAlphanumeric = new RegExp(/\w/i);
var regexEscapedHtml = /&(amp|quot|lt|gt|#39);/g;
var regexHasEscapedHtml = RegExp(regexEscapedHtml.source);
function unescape(value) {
    return value && regexHasEscapedHtml.test(value) ? value.replace(regexEscapedHtml, function(character) {
        return htmlEscapes[character];
    }) : value;
}
function isPartHighlighted(parts, i) {
    var _parts, _parts2;
    var current = parts[i];
    var isNextHighlighted = ((_parts = parts[i + 1]) === null || _parts === void 0 ? void 0 : _parts.isHighlighted) || true;
    var isPreviousHighlighted = ((_parts2 = parts[i - 1]) === null || _parts2 === void 0 ? void 0 : _parts2.isHighlighted) || true;
    if (!hasAlphanumeric.test(unescape(current.value)) && isPreviousHighlighted === isNextHighlighted) return isPreviousHighlighted;
    return current.isHighlighted;
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"hDmRU":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "parseAlgoliaHitReverseSnippet", ()=>parseAlgoliaHitReverseSnippet
);
var _parseAlgoliaHitSnippet = require("./parseAlgoliaHitSnippet");
var _reverseHighlightedParts = require("./reverseHighlightedParts");
function parseAlgoliaHitReverseSnippet(props) {
    return _reverseHighlightedParts.reverseHighlightedParts(_parseAlgoliaHitSnippet.parseAlgoliaHitSnippet(props));
}

},{"./parseAlgoliaHitSnippet":"e5gtH","./reverseHighlightedParts":"lnI4J","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"e5gtH":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "parseAlgoliaHitSnippet", ()=>parseAlgoliaHitSnippet
);
var _autocompleteShared = require("@algolia/autocomplete-shared");
var _parseAttribute = require("./parseAttribute");
function _toConsumableArray(arr) {
    return _arrayWithoutHoles(arr) || _iterableToArray(arr) || _unsupportedIterableToArray(arr) || _nonIterableSpread();
}
function _nonIterableSpread() {
    throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
}
function _unsupportedIterableToArray(o, minLen) {
    if (!o) return;
    if (typeof o === "string") return _arrayLikeToArray(o, minLen);
    var n = Object.prototype.toString.call(o).slice(8, -1);
    if (n === "Object" && o.constructor) n = o.constructor.name;
    if (n === "Map" || n === "Set") return Array.from(o);
    if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen);
}
function _iterableToArray(iter) {
    if (typeof Symbol !== "undefined" && iter[Symbol.iterator] != null || iter["@@iterator"] != null) return Array.from(iter);
}
function _arrayWithoutHoles(arr) {
    if (Array.isArray(arr)) return _arrayLikeToArray(arr);
}
function _arrayLikeToArray(arr, len) {
    if (len == null || len > arr.length) len = arr.length;
    for(var i = 0, arr2 = new Array(len); i < len; i++)arr2[i] = arr[i];
    return arr2;
}
function parseAlgoliaHitSnippet(_ref) {
    var hit = _ref.hit, attribute = _ref.attribute;
    var path = Array.isArray(attribute) ? attribute : [
        attribute
    ];
    var highlightedValue = _autocompleteShared.getAttributeValueByPath(hit, [
        '_snippetResult'
    ].concat(_toConsumableArray(path), [
        'value'
    ]));
    if (typeof highlightedValue !== 'string') {
        _autocompleteShared.warn(false, "The attribute \"".concat(path.join('.'), "\" described by the path ").concat(JSON.stringify(path), " does not exist on the hit. Did you set it in `attributesToSnippet`?") + '\nSee https://www.algolia.com/doc/api-reference/api-parameters/attributesToSnippet/');
        highlightedValue = _autocompleteShared.getAttributeValueByPath(hit, path) || '';
    }
    return _parseAttribute.parseAttribute({
        highlightedValue: highlightedValue
    });
}

},{"@algolia/autocomplete-shared":"59T59","./parseAttribute":"3rO3A","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"cZ7Rk":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"a8CYC":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _createRequester = require("./createRequester");
parcelHelpers.exportAll(_createRequester, exports);
var _getAlgoliaFacets = require("./getAlgoliaFacets");
parcelHelpers.exportAll(_getAlgoliaFacets, exports);
var _getAlgoliaResults = require("./getAlgoliaResults");
parcelHelpers.exportAll(_getAlgoliaResults, exports);

},{"./createRequester":"l5lzp","./getAlgoliaFacets":"djZ2z","./getAlgoliaResults":"4sIr0","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"l5lzp":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "createRequester", ()=>createRequester
);
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        enumerableOnly && (symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        })), keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = null != arguments[i] ? arguments[i] : {};
        i % 2 ? ownKeys(Object(source), !0).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
function createRequester(fetcher, requesterId) {
    function execute(fetcherParams) {
        return fetcher({
            searchClient: fetcherParams.searchClient,
            queries: fetcherParams.requests.map(function(x) {
                return x.query;
            })
        }).then(function(responses) {
            return responses.map(function(response, index) {
                var _fetcherParams$reques = fetcherParams.requests[index], sourceId = _fetcherParams$reques.sourceId, transformResponse = _fetcherParams$reques.transformResponse;
                return {
                    items: response,
                    sourceId: sourceId,
                    transformResponse: transformResponse
                };
            });
        });
    }
    return function createSpecifiedRequester(requesterParams) {
        return function requester(requestParams) {
            return _objectSpread(_objectSpread({
                requesterId: requesterId,
                execute: execute
            }, requesterParams), requestParams);
        };
    };
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"djZ2z":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/**
 * Retrieves Algolia facet hits from multiple indices.
 */ parcelHelpers.export(exports, "getAlgoliaFacets", ()=>getAlgoliaFacets
);
var _createAlgoliaRequester = require("./createAlgoliaRequester");
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        enumerableOnly && (symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        })), keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = null != arguments[i] ? arguments[i] : {};
        i % 2 ? ownKeys(Object(source), !0).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
function getAlgoliaFacets(requestParams) {
    var requester = _createAlgoliaRequester.createAlgoliaRequester({
        transformResponse: function transformResponse(response) {
            return response.facetHits;
        }
    });
    var queries = requestParams.queries.map(function(query) {
        return _objectSpread(_objectSpread({}, query), {}, {
            type: 'facet'
        });
    });
    return requester(_objectSpread(_objectSpread({}, requestParams), {}, {
        queries: queries
    }));
}

},{"./createAlgoliaRequester":"32HWJ","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"32HWJ":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "createAlgoliaRequester", ()=>createAlgoliaRequester
);
var _search = require("../search");
var _createRequester = require("./createRequester");
var createAlgoliaRequester = _createRequester.createRequester(_search.fetchAlgoliaResults, 'algolia');

},{"../search":"gwNho","./createRequester":"l5lzp","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"gwNho":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _fetchAlgoliaResults = require("./fetchAlgoliaResults");
parcelHelpers.exportAll(_fetchAlgoliaResults, exports);

},{"./fetchAlgoliaResults":"51Rml","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"51Rml":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "fetchAlgoliaResults", ()=>fetchAlgoliaResults
);
var _autocompleteShared = require("@algolia/autocomplete-shared");
var _constants = require("../constants");
var _excluded = [
    "params"
];
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        enumerableOnly && (symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        })), keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = null != arguments[i] ? arguments[i] : {};
        i % 2 ? ownKeys(Object(source), !0).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
function _objectWithoutProperties(source, excluded) {
    if (source == null) return {};
    var target = _objectWithoutPropertiesLoose(source, excluded);
    var key, i;
    if (Object.getOwnPropertySymbols) {
        var sourceSymbolKeys = Object.getOwnPropertySymbols(source);
        for(i = 0; i < sourceSymbolKeys.length; i++){
            key = sourceSymbolKeys[i];
            if (excluded.indexOf(key) >= 0) continue;
            if (!Object.prototype.propertyIsEnumerable.call(source, key)) continue;
            target[key] = source[key];
        }
    }
    return target;
}
function _objectWithoutPropertiesLoose(source, excluded) {
    if (source == null) return {};
    var target = {};
    var sourceKeys = Object.keys(source);
    var key, i;
    for(i = 0; i < sourceKeys.length; i++){
        key = sourceKeys[i];
        if (excluded.indexOf(key) >= 0) continue;
        target[key] = source[key];
    }
    return target;
}
function _toConsumableArray(arr) {
    return _arrayWithoutHoles(arr) || _iterableToArray(arr) || _unsupportedIterableToArray(arr) || _nonIterableSpread();
}
function _nonIterableSpread() {
    throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
}
function _unsupportedIterableToArray(o, minLen) {
    if (!o) return;
    if (typeof o === "string") return _arrayLikeToArray(o, minLen);
    var n = Object.prototype.toString.call(o).slice(8, -1);
    if (n === "Object" && o.constructor) n = o.constructor.name;
    if (n === "Map" || n === "Set") return Array.from(o);
    if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen);
}
function _iterableToArray(iter) {
    if (typeof Symbol !== "undefined" && iter[Symbol.iterator] != null || iter["@@iterator"] != null) return Array.from(iter);
}
function _arrayWithoutHoles(arr) {
    if (Array.isArray(arr)) return _arrayLikeToArray(arr);
}
function _arrayLikeToArray(arr, len) {
    if (len == null || len > arr.length) len = arr.length;
    for(var i = 0, arr2 = new Array(len); i < len; i++)arr2[i] = arr[i];
    return arr2;
}
function fetchAlgoliaResults(_ref) {
    var searchClient = _ref.searchClient, queries = _ref.queries, _ref$userAgents = _ref.userAgents, userAgents = _ref$userAgents === void 0 ? [] : _ref$userAgents;
    if (typeof searchClient.addAlgoliaAgent === 'function') {
        var algoliaAgents = [].concat(_toConsumableArray(_autocompleteShared.userAgents), _toConsumableArray(userAgents));
        algoliaAgents.forEach(function(_ref2) {
            var segment = _ref2.segment, version = _ref2.version;
            searchClient.addAlgoliaAgent(segment, version);
        });
    }
    return searchClient.search(queries.map(function(searchParameters) {
        var params = searchParameters.params, headers = _objectWithoutProperties(searchParameters, _excluded);
        return _objectSpread(_objectSpread({}, headers), {}, {
            params: _objectSpread({
                hitsPerPage: 5,
                highlightPreTag: _constants.HIGHLIGHT_PRE_TAG,
                highlightPostTag: _constants.HIGHLIGHT_POST_TAG
            }, params)
        });
    })).then(function(response) {
        return response.results;
    });
}

},{"@algolia/autocomplete-shared":"59T59","../constants":"hThxU","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"4sIr0":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "getAlgoliaResults", ()=>getAlgoliaResults
);
var _createAlgoliaRequester = require("./createAlgoliaRequester");
var getAlgoliaResults = _createAlgoliaRequester.createAlgoliaRequester({
    transformResponse: function transformResponse(response) {
        return response.hits;
    }
});

},{"./createAlgoliaRequester":"32HWJ","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"b85bW":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "createReverseHighlightComponent", ()=>createReverseHighlightComponent
);
var _autocompletePresetAlgolia = require("@algolia/autocomplete-preset-algolia");
function createReverseHighlightComponent(_ref) {
    var createElement = _ref.createElement, Fragment = _ref.Fragment;
    function ReverseHighlight(_ref2) {
        var hit = _ref2.hit, attribute = _ref2.attribute, _ref2$tagName = _ref2.tagName, tagName = _ref2$tagName === void 0 ? 'mark' : _ref2$tagName;
        return createElement(Fragment, {}, _autocompletePresetAlgolia.parseAlgoliaHitReverseHighlight({
            hit: hit,
            attribute: attribute
        }).map(function(x, index) {
            return x.isHighlighted ? createElement(tagName, {
                key: index
            }, x.value) : x.value;
        }));
    }
    ReverseHighlight.__autocomplete_componentName = 'ReverseHighlight';
    return ReverseHighlight;
}

},{"@algolia/autocomplete-preset-algolia":"lp3lw","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"fNt5i":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "createReverseSnippetComponent", ()=>createReverseSnippetComponent
);
var _autocompletePresetAlgolia = require("@algolia/autocomplete-preset-algolia");
function createReverseSnippetComponent(_ref) {
    var createElement = _ref.createElement, Fragment = _ref.Fragment;
    function ReverseSnippet(_ref2) {
        var hit = _ref2.hit, attribute = _ref2.attribute, _ref2$tagName = _ref2.tagName, tagName = _ref2$tagName === void 0 ? 'mark' : _ref2$tagName;
        return createElement(Fragment, {}, _autocompletePresetAlgolia.parseAlgoliaHitReverseSnippet({
            hit: hit,
            attribute: attribute
        }).map(function(x, index) {
            return x.isHighlighted ? createElement(tagName, {
                key: index
            }, x.value) : x.value;
        }));
    }
    ReverseSnippet.__autocomplete_componentName = 'ReverseSnippet';
    return ReverseSnippet;
}

},{"@algolia/autocomplete-preset-algolia":"lp3lw","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"bThlx":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "createSnippetComponent", ()=>createSnippetComponent
);
var _autocompletePresetAlgolia = require("@algolia/autocomplete-preset-algolia");
function createSnippetComponent(_ref) {
    var createElement = _ref.createElement, Fragment = _ref.Fragment;
    function Snippet(_ref2) {
        var hit = _ref2.hit, attribute = _ref2.attribute, _ref2$tagName = _ref2.tagName, tagName = _ref2$tagName === void 0 ? 'mark' : _ref2$tagName;
        return createElement(Fragment, {}, _autocompletePresetAlgolia.parseAlgoliaHitSnippet({
            hit: hit,
            attribute: attribute
        }).map(function(x, index) {
            return x.isHighlighted ? createElement(tagName, {
                key: index
            }, x.value) : x.value;
        }));
    }
    Snippet.__autocomplete_componentName = 'Snippet';
    return Snippet;
}

},{"@algolia/autocomplete-preset-algolia":"lp3lw","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"aVhxY":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "getPanelPlacementStyle", ()=>getPanelPlacementStyle
);
function getPanelPlacementStyle(_ref) {
    var panelPlacement = _ref.panelPlacement, container = _ref.container, form = _ref.form, environment = _ref.environment;
    var containerRect = container.getBoundingClientRect(); // Some browsers have specificities to retrieve the document scroll position.
    // See https://stackoverflow.com/a/28633515/9940315
    var scrollTop = environment.pageYOffset || environment.document.documentElement.scrollTop || environment.document.body.scrollTop || 0;
    var top = scrollTop + containerRect.top + containerRect.height;
    switch(panelPlacement){
        case 'start':
            return {
                top: top,
                left: containerRect.left
            };
        case 'end':
            return {
                top: top,
                right: environment.document.documentElement.clientWidth - (containerRect.left + containerRect.width)
            };
        case 'full-width':
            return {
                top: top,
                left: 0,
                right: 0,
                width: 'unset',
                maxWidth: 'unset'
            };
        case 'input-wrapper-width':
            var formRect = form.getBoundingClientRect();
            return {
                top: top,
                left: formRect.left,
                right: environment.document.documentElement.clientWidth - (formRect.left + formRect.width),
                width: 'unset',
                maxWidth: 'unset'
            };
        default:
            throw new Error("[Autocomplete] The `panelPlacement` value ".concat(JSON.stringify(panelPlacement), " is not valid."));
    }
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"45iVn":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "renderSearchBox", ()=>renderSearchBox
);
parcelHelpers.export(exports, "renderPanel", ()=>renderPanel
);
/** @jsx renderer.createElement */ var _utils = require("./utils");
function _extends() {
    _extends = Object.assign || function(target) {
        for(var i = 1; i < arguments.length; i++){
            var source = arguments[i];
            for(var key in source)if (Object.prototype.hasOwnProperty.call(source, key)) target[key] = source[key];
        }
        return target;
    };
    return _extends.apply(this, arguments);
}
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        enumerableOnly && (symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        })), keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = null != arguments[i] ? arguments[i] : {};
        i % 2 ? ownKeys(Object(source), !0).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
function renderSearchBox(_ref) {
    var autocomplete = _ref.autocomplete, autocompleteScopeApi = _ref.autocompleteScopeApi, dom = _ref.dom, propGetters = _ref.propGetters, state = _ref.state;
    _utils.setPropertiesWithoutEvents(dom.root, propGetters.getRootProps(_objectSpread({
        state: state,
        props: autocomplete.getRootProps({})
    }, autocompleteScopeApi)));
    _utils.setPropertiesWithoutEvents(dom.input, propGetters.getInputProps(_objectSpread({
        state: state,
        props: autocomplete.getInputProps({
            inputElement: dom.input
        }),
        inputElement: dom.input
    }, autocompleteScopeApi)));
    _utils.setProperties(dom.label, {
        hidden: state.status === 'stalled'
    });
    _utils.setProperties(dom.loadingIndicator, {
        hidden: state.status !== 'stalled'
    });
    _utils.setProperties(dom.clearButton, {
        hidden: !state.query
    });
}
function renderPanel(render, _ref2) {
    var autocomplete = _ref2.autocomplete, autocompleteScopeApi = _ref2.autocompleteScopeApi, classNames = _ref2.classNames, html = _ref2.html, dom = _ref2.dom, panelContainer = _ref2.panelContainer, propGetters = _ref2.propGetters, state = _ref2.state, components = _ref2.components, renderer = _ref2.renderer;
    if (!state.isOpen) {
        if (panelContainer.contains(dom.panel)) panelContainer.removeChild(dom.panel);
        return;
    } // We add the panel element to the DOM when it's not yet appended and that the
    // items are fetched.
    if (!panelContainer.contains(dom.panel) && state.status !== 'loading') panelContainer.appendChild(dom.panel);
    dom.panel.classList.toggle('aa-Panel--stalled', state.status === 'stalled');
    var sections = state.collections.filter(function(_ref3) {
        var source = _ref3.source, items = _ref3.items;
        return source.templates.noResults || items.length > 0;
    }).map(function(_ref4, sourceIndex) {
        var source = _ref4.source, items = _ref4.items;
        return renderer.createElement("section", {
            key: sourceIndex,
            className: classNames.source,
            "data-autocomplete-source-id": source.sourceId
        }, source.templates.header && renderer.createElement("div", {
            className: classNames.sourceHeader
        }, source.templates.header({
            components: components,
            createElement: renderer.createElement,
            Fragment: renderer.Fragment,
            items: items,
            source: source,
            state: state,
            html: html
        })), source.templates.noResults && items.length === 0 ? renderer.createElement("div", {
            className: classNames.sourceNoResults
        }, source.templates.noResults({
            components: components,
            createElement: renderer.createElement,
            Fragment: renderer.Fragment,
            source: source,
            state: state,
            html: html
        })) : renderer.createElement("ul", _extends({
            className: classNames.list
        }, propGetters.getListProps(_objectSpread({
            state: state,
            props: autocomplete.getListProps({})
        }, autocompleteScopeApi))), items.map(function(item) {
            var itemProps = autocomplete.getItemProps({
                item: item,
                source: source
            });
            return renderer.createElement("li", _extends({
                key: itemProps.id,
                className: classNames.item
            }, propGetters.getItemProps(_objectSpread({
                state: state,
                props: itemProps
            }, autocompleteScopeApi))), source.templates.item({
                components: components,
                createElement: renderer.createElement,
                Fragment: renderer.Fragment,
                item: item,
                state: state,
                html: html
            }));
        })), source.templates.footer && renderer.createElement("div", {
            className: classNames.sourceFooter
        }, source.templates.footer({
            components: components,
            createElement: renderer.createElement,
            Fragment: renderer.Fragment,
            items: items,
            source: source,
            state: state,
            html: html
        })));
    });
    var children = renderer.createElement(renderer.Fragment, null, renderer.createElement("div", {
        className: classNames.panelLayout
    }, sections), renderer.createElement("div", {
        className: "aa-GradientBottom"
    }));
    var elements = sections.reduce(function(acc, current) {
        acc[current.props['data-autocomplete-source-id']] = current;
        return acc;
    }, {});
    render(_objectSpread(_objectSpread({
        children: children,
        state: state,
        sections: sections,
        elements: elements
    }, renderer), {}, {
        components: components,
        html: html
    }, autocompleteScopeApi), dom.panel);
}

},{"./utils":"fKU1x","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"hqEHF":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "userAgents", ()=>userAgents
);
var _autocompleteShared = require("@algolia/autocomplete-shared");
var userAgents = [
    {
        segment: 'autocomplete-js',
        version: _autocompleteShared.version
    }
];

},{"@algolia/autocomplete-shared":"59T59","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"24Y2H":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _getAlgoliaFacets = require("./getAlgoliaFacets");
parcelHelpers.exportAll(_getAlgoliaFacets, exports);
var _getAlgoliaResults = require("./getAlgoliaResults");
parcelHelpers.exportAll(_getAlgoliaResults, exports);

},{"./getAlgoliaFacets":"8NocL","./getAlgoliaResults":"4tl7s","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"8NocL":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/**
 * Retrieves Algolia facet hits from multiple indices.
 */ parcelHelpers.export(exports, "getAlgoliaFacets", ()=>getAlgoliaFacets
);
var _createAlgoliaRequester = require("./createAlgoliaRequester");
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        enumerableOnly && (symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        })), keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = null != arguments[i] ? arguments[i] : {};
        i % 2 ? ownKeys(Object(source), !0).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
function getAlgoliaFacets(requestParams) {
    var requester = _createAlgoliaRequester.createAlgoliaRequester({
        transformResponse: function transformResponse(response) {
            return response.facetHits;
        }
    });
    var queries = requestParams.queries.map(function(query) {
        return _objectSpread(_objectSpread({}, query), {}, {
            type: 'facet'
        });
    });
    return requester(_objectSpread(_objectSpread({}, requestParams), {}, {
        queries: queries
    }));
}

},{"./createAlgoliaRequester":"fsJ9y","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"fsJ9y":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "createAlgoliaRequester", ()=>createAlgoliaRequester
);
var _autocompletePresetAlgolia = require("@algolia/autocomplete-preset-algolia");
var _userAgents = require("../userAgents");
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        enumerableOnly && (symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        })), keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = null != arguments[i] ? arguments[i] : {};
        i % 2 ? ownKeys(Object(source), !0).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
var createAlgoliaRequester = _autocompletePresetAlgolia.createRequester(function(params) {
    return _autocompletePresetAlgolia.fetchAlgoliaResults(_objectSpread(_objectSpread({}, params), {}, {
        userAgents: _userAgents.userAgents
    }));
}, 'algolia');

},{"@algolia/autocomplete-preset-algolia":"lp3lw","../userAgents":"hqEHF","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"4tl7s":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "getAlgoliaResults", ()=>getAlgoliaResults
);
var _createAlgoliaRequester = require("./createAlgoliaRequester");
var getAlgoliaResults = _createAlgoliaRequester.createAlgoliaRequester({
    transformResponse: function transformResponse(response) {
        return response.hits;
    }
});

},{"./createAlgoliaRequester":"fsJ9y","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"8cK74":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _autocompleteApi = require("./AutocompleteApi");
parcelHelpers.exportAll(_autocompleteApi, exports);
var _autocompleteClassNames = require("./AutocompleteClassNames");
parcelHelpers.exportAll(_autocompleteClassNames, exports);
var _autocompleteCollection = require("./AutocompleteCollection");
parcelHelpers.exportAll(_autocompleteCollection, exports);
var _autocompleteComponents = require("./AutocompleteComponents");
parcelHelpers.exportAll(_autocompleteComponents, exports);
var _autocompleteDom = require("./AutocompleteDom");
parcelHelpers.exportAll(_autocompleteDom, exports);
var _autocompleteOptions = require("./AutocompleteOptions");
parcelHelpers.exportAll(_autocompleteOptions, exports);
var _autocompletePlugin = require("./AutocompletePlugin");
parcelHelpers.exportAll(_autocompletePlugin, exports);
var _autocompletePropGetters = require("./AutocompletePropGetters");
parcelHelpers.exportAll(_autocompletePropGetters, exports);
var _autocompleteRender = require("./AutocompleteRender");
parcelHelpers.exportAll(_autocompleteRender, exports);
var _autocompleteRenderer = require("./AutocompleteRenderer");
parcelHelpers.exportAll(_autocompleteRenderer, exports);
var _autocompleteSource = require("./AutocompleteSource");
parcelHelpers.exportAll(_autocompleteSource, exports);
var _autocompleteState = require("./AutocompleteState");
parcelHelpers.exportAll(_autocompleteState, exports);
var _autocompleteTranslations = require("./AutocompleteTranslations");
parcelHelpers.exportAll(_autocompleteTranslations, exports);
var _highlightHitParams = require("./HighlightHitParams");
parcelHelpers.exportAll(_highlightHitParams, exports);

},{"./AutocompleteApi":"9PH6B","./AutocompleteClassNames":"1p0cB","./AutocompleteCollection":"5bmmV","./AutocompleteComponents":"cARCb","./AutocompleteDom":"duZcj","./AutocompleteOptions":"a34Ht","./AutocompletePlugin":"hL4WT","./AutocompletePropGetters":"2ubDr","./AutocompleteRender":"8ciX1","./AutocompleteRenderer":"3KaZb","./AutocompleteSource":"29qGC","./AutocompleteState":"imzQ1","./AutocompleteTranslations":"lG9me","./HighlightHitParams":"hgzci","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"9PH6B":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"1p0cB":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"5bmmV":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"cARCb":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"duZcj":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"a34Ht":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"hL4WT":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"2ubDr":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"8ciX1":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"3KaZb":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"29qGC":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"imzQ1":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"lG9me":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"hgzci":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"fWJNO":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "connectClearRefinements", ()=>_connectClearRefinementsDefault.default
);
parcelHelpers.export(exports, "connectCurrentRefinements", ()=>_connectCurrentRefinementsDefault.default
);
parcelHelpers.export(exports, "connectHierarchicalMenu", ()=>_connectHierarchicalMenuDefault.default
);
parcelHelpers.export(exports, "connectHits", ()=>_connectHitsDefault.default
);
parcelHelpers.export(exports, "connectHitsWithInsights", ()=>_connectHitsWithInsightsDefault.default
);
parcelHelpers.export(exports, "connectHitsPerPage", ()=>_connectHitsPerPageDefault.default
);
parcelHelpers.export(exports, "connectInfiniteHits", ()=>_connectInfiniteHitsDefault.default
);
parcelHelpers.export(exports, "connectInfiniteHitsWithInsights", ()=>_connectInfiniteHitsWithInsightsDefault.default
);
parcelHelpers.export(exports, "connectMenu", ()=>_connectMenuDefault.default
);
parcelHelpers.export(exports, "connectNumericMenu", ()=>_connectNumericMenuDefault.default
);
parcelHelpers.export(exports, "connectPagination", ()=>_connectPaginationDefault.default
);
parcelHelpers.export(exports, "connectRange", ()=>_connectRangeDefault.default
);
parcelHelpers.export(exports, "connectRefinementList", ()=>_connectRefinementListDefault.default
);
parcelHelpers.export(exports, "connectSearchBox", ()=>_connectSearchBoxDefault.default
);
parcelHelpers.export(exports, "connectSortBy", ()=>_connectSortByDefault.default
);
parcelHelpers.export(exports, "connectRatingMenu", ()=>_connectRatingMenuDefault.default
);
parcelHelpers.export(exports, "connectStats", ()=>_connectStatsDefault.default
);
parcelHelpers.export(exports, "connectToggleRefinement", ()=>_connectToggleRefinementDefault.default
);
parcelHelpers.export(exports, "connectBreadcrumb", ()=>_connectBreadcrumbDefault.default
);
parcelHelpers.export(exports, "connectGeoSearch", ()=>_connectGeoSearchDefault.default
);
parcelHelpers.export(exports, "connectPoweredBy", ()=>_connectPoweredByDefault.default
);
parcelHelpers.export(exports, "connectConfigure", ()=>_connectConfigureDefault.default
);
parcelHelpers.export(exports, "EXPERIMENTAL_connectConfigureRelatedItems", ()=>_connectConfigureRelatedItemsDefault.default
);
parcelHelpers.export(exports, "connectAutocomplete", ()=>_connectAutocompleteDefault.default
);
parcelHelpers.export(exports, "connectQueryRules", ()=>_connectQueryRulesDefault.default
);
parcelHelpers.export(exports, "connectVoiceSearch", ()=>_connectVoiceSearchDefault.default
);
parcelHelpers.export(exports, "EXPERIMENTAL_connectAnswers", ()=>_connectAnswersDefault.default
);
parcelHelpers.export(exports, "connectRelevantSort", ()=>_connectRelevantSortDefault.default
);
parcelHelpers.export(exports, "EXPERIMENTAL_connectDynamicWidgets", ()=>_connectDynamicWidgetsDefault.default
);
var _connectClearRefinements = require("./clear-refinements/connectClearRefinements");
var _connectClearRefinementsDefault = parcelHelpers.interopDefault(_connectClearRefinements);
var _connectCurrentRefinements = require("./current-refinements/connectCurrentRefinements");
var _connectCurrentRefinementsDefault = parcelHelpers.interopDefault(_connectCurrentRefinements);
var _connectHierarchicalMenu = require("./hierarchical-menu/connectHierarchicalMenu");
var _connectHierarchicalMenuDefault = parcelHelpers.interopDefault(_connectHierarchicalMenu);
var _connectHits = require("./hits/connectHits");
var _connectHitsDefault = parcelHelpers.interopDefault(_connectHits);
var _connectHitsWithInsights = require("./hits/connectHitsWithInsights");
var _connectHitsWithInsightsDefault = parcelHelpers.interopDefault(_connectHitsWithInsights);
var _connectHitsPerPage = require("./hits-per-page/connectHitsPerPage");
var _connectHitsPerPageDefault = parcelHelpers.interopDefault(_connectHitsPerPage);
var _connectInfiniteHits = require("./infinite-hits/connectInfiniteHits");
var _connectInfiniteHitsDefault = parcelHelpers.interopDefault(_connectInfiniteHits);
var _connectInfiniteHitsWithInsights = require("./infinite-hits/connectInfiniteHitsWithInsights");
var _connectInfiniteHitsWithInsightsDefault = parcelHelpers.interopDefault(_connectInfiniteHitsWithInsights);
var _connectMenu = require("./menu/connectMenu");
var _connectMenuDefault = parcelHelpers.interopDefault(_connectMenu);
var _connectNumericMenu = require("./numeric-menu/connectNumericMenu");
var _connectNumericMenuDefault = parcelHelpers.interopDefault(_connectNumericMenu);
var _connectPagination = require("./pagination/connectPagination");
var _connectPaginationDefault = parcelHelpers.interopDefault(_connectPagination);
var _connectRange = require("./range/connectRange");
var _connectRangeDefault = parcelHelpers.interopDefault(_connectRange);
var _connectRefinementList = require("./refinement-list/connectRefinementList");
var _connectRefinementListDefault = parcelHelpers.interopDefault(_connectRefinementList);
var _connectSearchBox = require("./search-box/connectSearchBox");
var _connectSearchBoxDefault = parcelHelpers.interopDefault(_connectSearchBox);
var _connectSortBy = require("./sort-by/connectSortBy");
var _connectSortByDefault = parcelHelpers.interopDefault(_connectSortBy);
var _connectRatingMenu = require("./rating-menu/connectRatingMenu");
var _connectRatingMenuDefault = parcelHelpers.interopDefault(_connectRatingMenu);
var _connectStats = require("./stats/connectStats");
var _connectStatsDefault = parcelHelpers.interopDefault(_connectStats);
var _connectToggleRefinement = require("./toggle-refinement/connectToggleRefinement");
var _connectToggleRefinementDefault = parcelHelpers.interopDefault(_connectToggleRefinement);
var _connectBreadcrumb = require("./breadcrumb/connectBreadcrumb");
var _connectBreadcrumbDefault = parcelHelpers.interopDefault(_connectBreadcrumb);
var _connectGeoSearch = require("./geo-search/connectGeoSearch");
var _connectGeoSearchDefault = parcelHelpers.interopDefault(_connectGeoSearch);
var _connectPoweredBy = require("./powered-by/connectPoweredBy");
var _connectPoweredByDefault = parcelHelpers.interopDefault(_connectPoweredBy);
var _connectConfigure = require("./configure/connectConfigure");
var _connectConfigureDefault = parcelHelpers.interopDefault(_connectConfigure);
var _connectConfigureRelatedItems = require("./configure-related-items/connectConfigureRelatedItems");
var _connectConfigureRelatedItemsDefault = parcelHelpers.interopDefault(_connectConfigureRelatedItems);
var _connectAutocomplete = require("./autocomplete/connectAutocomplete");
var _connectAutocompleteDefault = parcelHelpers.interopDefault(_connectAutocomplete);
var _connectQueryRules = require("./query-rules/connectQueryRules");
var _connectQueryRulesDefault = parcelHelpers.interopDefault(_connectQueryRules);
var _connectVoiceSearch = require("./voice-search/connectVoiceSearch");
var _connectVoiceSearchDefault = parcelHelpers.interopDefault(_connectVoiceSearch);
var _connectAnswers = require("./answers/connectAnswers");
var _connectAnswersDefault = parcelHelpers.interopDefault(_connectAnswers);
var _connectRelevantSort = require("./relevant-sort/connectRelevantSort");
var _connectRelevantSortDefault = parcelHelpers.interopDefault(_connectRelevantSort);
var _connectDynamicWidgets = require("./dynamic-widgets/connectDynamicWidgets");
var _connectDynamicWidgetsDefault = parcelHelpers.interopDefault(_connectDynamicWidgets);

},{"./clear-refinements/connectClearRefinements":false,"./current-refinements/connectCurrentRefinements":false,"./hierarchical-menu/connectHierarchicalMenu":false,"./hits/connectHits":"b5DNx","./hits/connectHitsWithInsights":false,"./hits-per-page/connectHitsPerPage":false,"./infinite-hits/connectInfiniteHits":false,"./infinite-hits/connectInfiniteHitsWithInsights":false,"./menu/connectMenu":false,"./numeric-menu/connectNumericMenu":false,"./pagination/connectPagination":"bHouJ","./range/connectRange":"abXn7","./refinement-list/connectRefinementList":"kkKYv","./search-box/connectSearchBox":"kqCmi","./sort-by/connectSortBy":"3pFgJ","./rating-menu/connectRatingMenu":false,"./stats/connectStats":false,"./toggle-refinement/connectToggleRefinement":false,"./breadcrumb/connectBreadcrumb":false,"./geo-search/connectGeoSearch":false,"./powered-by/connectPoweredBy":false,"./configure/connectConfigure":"lvgHS","./configure-related-items/connectConfigureRelatedItems":false,"./autocomplete/connectAutocomplete":false,"./query-rules/connectQueryRules":false,"./voice-search/connectVoiceSearch":false,"./answers/connectAnswers":false,"./relevant-sort/connectRelevantSort":false,"./dynamic-widgets/connectDynamicWidgets":false,"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"b5DNx":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _utils = require("../../lib/utils");
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        if (enumerableOnly) symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        });
        keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = arguments[i] != null ? arguments[i] : {};
        if (i % 2) ownKeys(Object(source), true).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        });
        else if (Object.getOwnPropertyDescriptors) Object.defineProperties(target, Object.getOwnPropertyDescriptors(source));
        else ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
var withUsage = _utils.createDocumentationMessageGenerator({
    name: 'hits',
    connector: true
});
var connectHits = function connectHits(renderFn) {
    var unmountFn = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : _utils.noop;
    _utils.checkRendering(renderFn, withUsage());
    return function(widgetParams) {
        var _ref = widgetParams || {}, _ref$escapeHTML = _ref.escapeHTML, escapeHTML = _ref$escapeHTML === void 0 ? true : _ref$escapeHTML, _ref$transformItems = _ref.transformItems, transformItems = _ref$transformItems === void 0 ? function(items) {
            return items;
        } : _ref$transformItems;
        var sendEvent;
        var bindEvent;
        return {
            $$type: 'ais.hits',
            init: function init(initOptions) {
                renderFn(_objectSpread(_objectSpread({}, this.getWidgetRenderState(initOptions)), {}, {
                    instantSearchInstance: initOptions.instantSearchInstance
                }), true);
            },
            render: function render(renderOptions) {
                var renderState = this.getWidgetRenderState(renderOptions);
                renderState.sendEvent('view', renderState.hits);
                renderFn(_objectSpread(_objectSpread({}, renderState), {}, {
                    instantSearchInstance: renderOptions.instantSearchInstance
                }), false);
            },
            getRenderState: function getRenderState(renderState, renderOptions) {
                return _objectSpread(_objectSpread({}, renderState), {}, {
                    hits: this.getWidgetRenderState(renderOptions)
                });
            },
            getWidgetRenderState: function getWidgetRenderState(_ref2) {
                var results = _ref2.results, helper = _ref2.helper, instantSearchInstance = _ref2.instantSearchInstance;
                if (!sendEvent) sendEvent = _utils.createSendEventForHits({
                    instantSearchInstance: instantSearchInstance,
                    index: helper.getIndex(),
                    widgetType: this.$$type
                });
                if (!bindEvent) bindEvent = _utils.createBindEventForHits({
                    index: helper.getIndex(),
                    widgetType: this.$$type
                });
                if (!results) return {
                    hits: [],
                    results: undefined,
                    sendEvent: sendEvent,
                    bindEvent: bindEvent,
                    widgetParams: widgetParams
                };
                if (escapeHTML && results.hits.length > 0) results.hits = _utils.escapeHits(results.hits);
                var initialEscaped = results.hits.__escaped;
                results.hits = _utils.addAbsolutePosition(results.hits, results.page, results.hitsPerPage);
                results.hits = _utils.addQueryID(results.hits, results.queryID);
                results.hits = transformItems(results.hits); // Make sure the escaped tag stays, even after mapping over the hits.
                // This prevents the hits from being double-escaped if there are multiple
                // hits widgets mounted on the page.
                results.hits.__escaped = initialEscaped;
                return {
                    hits: results.hits,
                    results: results,
                    sendEvent: sendEvent,
                    bindEvent: bindEvent,
                    widgetParams: widgetParams
                };
            },
            dispose: function dispose(_ref3) {
                var state = _ref3.state;
                unmountFn();
                if (!escapeHTML) return state;
                return state.setQueryParameters(Object.keys(_utils.TAG_PLACEHOLDER).reduce(function(acc, key) {
                    return _objectSpread(_objectSpread({}, acc), {}, _defineProperty({}, key, undefined));
                }, {}));
            },
            getWidgetSearchParameters: function getWidgetSearchParameters(state) {
                if (!escapeHTML) return state;
                return state.setQueryParameters(_utils.TAG_PLACEHOLDER);
            }
        };
    };
};
exports.default = connectHits;

},{"../../lib/utils":"etVYs","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"bHouJ":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _utils = require("../../lib/utils");
var _paginator = require("./Paginator");
var _paginatorDefault = parcelHelpers.interopDefault(_paginator);
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        if (enumerableOnly) symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        });
        keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = arguments[i] != null ? arguments[i] : {};
        if (i % 2) ownKeys(Object(source), true).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        });
        else if (Object.getOwnPropertyDescriptors) Object.defineProperties(target, Object.getOwnPropertyDescriptors(source));
        else ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
var withUsage = _utils.createDocumentationMessageGenerator({
    name: 'pagination',
    connector: true
});
/**
 * **Pagination** connector provides the logic to build a widget that will let the user
 * choose the current page of the results.
 *
 * When using the pagination with Algolia, you should be aware that the engine won't provide you pages
 * beyond the 1000th hits by default. You can find more information on the [Algolia documentation](https://www.algolia.com/doc/guides/searching/pagination/#pagination-limitations).
 */ var connectPagination = function connectPagination(renderFn) {
    var unmountFn = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : _utils.noop;
    _utils.checkRendering(renderFn, withUsage());
    return function(widgetParams) {
        var _ref = widgetParams || {}, totalPages = _ref.totalPages, _ref$padding = _ref.padding, padding = _ref$padding === void 0 ? 3 : _ref$padding;
        var pager = new _paginatorDefault.default({
            currentPage: 0,
            total: 0,
            padding: padding
        });
        var connectorState = {};
        function getMaxPage(_ref2) {
            var nbPages = _ref2.nbPages;
            return totalPages !== undefined ? Math.min(totalPages, nbPages) : nbPages;
        }
        return {
            $$type: 'ais.pagination',
            init: function init(initOptions) {
                var instantSearchInstance = initOptions.instantSearchInstance;
                renderFn(_objectSpread(_objectSpread({}, this.getWidgetRenderState(initOptions)), {}, {
                    instantSearchInstance: instantSearchInstance
                }), true);
            },
            render: function render(renderOptions) {
                var instantSearchInstance = renderOptions.instantSearchInstance;
                renderFn(_objectSpread(_objectSpread({}, this.getWidgetRenderState(renderOptions)), {}, {
                    instantSearchInstance: instantSearchInstance
                }), false);
            },
            dispose: function dispose(_ref3) {
                var state = _ref3.state;
                unmountFn();
                return state.setQueryParameter('page', undefined);
            },
            getWidgetUiState: function getWidgetUiState(uiState, _ref4) {
                var searchParameters = _ref4.searchParameters;
                var page = searchParameters.page || 0;
                if (!page) return uiState;
                return _objectSpread(_objectSpread({}, uiState), {}, {
                    page: page + 1
                });
            },
            getWidgetSearchParameters: function getWidgetSearchParameters(searchParameters, _ref5) {
                var uiState = _ref5.uiState;
                var page = uiState.page ? uiState.page - 1 : 0;
                return searchParameters.setQueryParameter('page', page);
            },
            getWidgetRenderState: function getWidgetRenderState(_ref6) {
                var results = _ref6.results, helper = _ref6.helper, createURL = _ref6.createURL;
                if (!connectorState.refine) connectorState.refine = function(page) {
                    helper.setPage(page);
                    helper.search();
                };
                if (!connectorState.createURL) connectorState.createURL = function(state) {
                    return function(page) {
                        return createURL(state.setPage(page));
                    };
                };
                var state1 = helper.state;
                var page1 = state1.page || 0;
                var nbPages = getMaxPage(results || {
                    nbPages: 0
                });
                pager.currentPage = page1;
                pager.total = nbPages;
                return {
                    createURL: connectorState.createURL(state1),
                    refine: connectorState.refine,
                    canRefine: nbPages > 1,
                    currentRefinement: page1,
                    nbHits: (results === null || results === void 0 ? void 0 : results.nbHits) || 0,
                    nbPages: nbPages,
                    pages: results ? pager.pages() : [],
                    isFirstPage: pager.isFirstPage(),
                    isLastPage: pager.isLastPage(),
                    widgetParams: widgetParams
                };
            },
            getRenderState: function getRenderState(renderState, renderOptions) {
                return _objectSpread(_objectSpread({}, renderState), {}, {
                    pagination: this.getWidgetRenderState(renderOptions)
                });
            }
        };
    };
};
exports.default = connectPagination;

},{"../../lib/utils":"etVYs","./Paginator":"4Vcbp","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"4Vcbp":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _utils = require("../../lib/utils");
function _classCallCheck(instance, Constructor) {
    if (!(instance instanceof Constructor)) throw new TypeError("Cannot call a class as a function");
}
function _defineProperties(target, props) {
    for(var i = 0; i < props.length; i++){
        var descriptor = props[i];
        descriptor.enumerable = descriptor.enumerable || false;
        descriptor.configurable = true;
        if ("value" in descriptor) descriptor.writable = true;
        Object.defineProperty(target, descriptor.key, descriptor);
    }
}
function _createClass(Constructor, protoProps, staticProps) {
    if (protoProps) _defineProperties(Constructor.prototype, protoProps);
    if (staticProps) _defineProperties(Constructor, staticProps);
    return Constructor;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
var Paginator = /*#__PURE__*/ function() {
    function Paginator1(params) {
        _classCallCheck(this, Paginator1);
        _defineProperty(this, "currentPage", void 0);
        _defineProperty(this, "total", void 0);
        _defineProperty(this, "padding", void 0);
        this.currentPage = params.currentPage;
        this.total = params.total;
        this.padding = params.padding;
    }
    _createClass(Paginator1, [
        {
            key: "pages",
            value: function pages() {
                var total = this.total, currentPage = this.currentPage, padding = this.padding;
                if (total === 0) return [
                    0
                ];
                var totalDisplayedPages = this.nbPagesDisplayed(padding, total);
                if (totalDisplayedPages === total) return _utils.range({
                    end: total
                });
                var paddingLeft = this.calculatePaddingLeft(currentPage, padding, total, totalDisplayedPages);
                var paddingRight = totalDisplayedPages - paddingLeft;
                var first = currentPage - paddingLeft;
                var last = currentPage + paddingRight;
                return _utils.range({
                    start: first,
                    end: last
                });
            }
        },
        {
            key: "nbPagesDisplayed",
            value: function nbPagesDisplayed(padding, total) {
                return Math.min(2 * padding + 1, total);
            }
        },
        {
            key: "calculatePaddingLeft",
            value: function calculatePaddingLeft(current, padding, total, totalDisplayedPages) {
                if (current <= padding) return current;
                if (current >= total - padding) return totalDisplayedPages - (total - current);
                return padding;
            }
        },
        {
            key: "isLastPage",
            value: function isLastPage() {
                return this.currentPage === this.total - 1 || this.total === 0;
            }
        },
        {
            key: "isFirstPage",
            value: function isFirstPage() {
                return this.currentPage === 0;
            }
        }
    ]);
    return Paginator1;
}();
exports.default = Paginator;

},{"../../lib/utils":"etVYs","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"abXn7":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _utils = require("../../lib/utils");
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        if (enumerableOnly) symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        });
        keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = arguments[i] != null ? arguments[i] : {};
        if (i % 2) ownKeys(Object(source), true).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        });
        else if (Object.getOwnPropertyDescriptors) Object.defineProperties(target, Object.getOwnPropertyDescriptors(source));
        else ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
function _slicedToArray(arr, i) {
    return _arrayWithHoles(arr) || _iterableToArrayLimit(arr, i) || _unsupportedIterableToArray(arr, i) || _nonIterableRest();
}
function _nonIterableRest() {
    throw new TypeError("Invalid attempt to destructure non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
}
function _unsupportedIterableToArray(o, minLen) {
    if (!o) return;
    if (typeof o === "string") return _arrayLikeToArray(o, minLen);
    var n = Object.prototype.toString.call(o).slice(8, -1);
    if (n === "Object" && o.constructor) n = o.constructor.name;
    if (n === "Map" || n === "Set") return Array.from(o);
    if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen);
}
function _arrayLikeToArray(arr, len) {
    if (len == null || len > arr.length) len = arr.length;
    for(var i = 0, arr2 = new Array(len); i < len; i++)arr2[i] = arr[i];
    return arr2;
}
function _iterableToArrayLimit(arr, i) {
    if (typeof Symbol === "undefined" || !(Symbol.iterator in Object(arr))) return;
    var _arr = [];
    var _n = true;
    var _d = false;
    var _e = undefined;
    try {
        for(var _i = arr[Symbol.iterator](), _s; !(_n = (_s = _i.next()).done); _n = true){
            _arr.push(_s.value);
            if (i && _arr.length === i) break;
        }
    } catch (err) {
        _d = true;
        _e = err;
    } finally{
        try {
            if (!_n && _i["return"] != null) _i["return"]();
        } finally{
            if (_d) throw _e;
        }
    }
    return _arr;
}
function _arrayWithHoles(arr) {
    if (Array.isArray(arr)) return arr;
}
var withUsage = _utils.createDocumentationMessageGenerator({
    name: 'range-input',
    connector: true
}, {
    name: 'range-slider',
    connector: true
});
var $$type = 'ais.range';
function toPrecision(_ref) {
    var min = _ref.min, max = _ref.max, precision = _ref.precision;
    var pow = Math.pow(10, precision);
    return {
        min: min ? Math.floor(min * pow) / pow : min,
        max: max ? Math.ceil(max * pow) / pow : max
    };
}
/**
 * **Range** connector provides the logic to create custom widget that will let
 * the user refine results using a numeric range.
 *
 * This connectors provides a `refine()` function that accepts bounds. It will also provide
 * information about the min and max bounds for the current result set.
 */ var connectRange = function connectRange(renderFn) {
    var unmountFn = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : _utils.noop;
    _utils.checkRendering(renderFn, withUsage());
    return function(widgetParams) {
        var _ref2 = widgetParams || {}, _ref2$attribute = _ref2.attribute, attribute = _ref2$attribute === void 0 ? '' : _ref2$attribute, minBound = _ref2.min, maxBound = _ref2.max, _ref2$precision = _ref2.precision, precision = _ref2$precision === void 0 ? 0 : _ref2$precision;
        if (!attribute) throw new Error(withUsage('The `attribute` option is required.'));
        if (_utils.isFiniteNumber(minBound) && _utils.isFiniteNumber(maxBound) && minBound > maxBound) throw new Error(withUsage("The `max` option can't be lower than `min`."));
        var formatToNumber = function formatToNumber(v) {
            return Number(Number(v).toFixed(precision));
        };
        var rangeFormatter = {
            from: function from(v) {
                return v.toLocaleString();
            },
            to: function to(v) {
                return formatToNumber(v).toLocaleString();
            }
        }; // eslint-disable-next-line complexity
        var getRefinedState = function getRefinedState(helper, currentRange, nextMin, nextMax) {
            var resolvedState = helper.state;
            var currentRangeMin = currentRange.min, currentRangeMax = currentRange.max;
            var _ref3 = resolvedState.getNumericRefinement(attribute, '>=') || [], _ref4 = _slicedToArray(_ref3, 1), min = _ref4[0];
            var _ref5 = resolvedState.getNumericRefinement(attribute, '<=') || [], _ref6 = _slicedToArray(_ref5, 1), max = _ref6[0];
            var isResetMin = nextMin === undefined || nextMin === '';
            var isResetMax = nextMax === undefined || nextMax === '';
            var _toPrecision = toPrecision({
                min: !isResetMin ? parseFloat(nextMin) : undefined,
                max: !isResetMax ? parseFloat(nextMax) : undefined,
                precision: precision
            }), nextMinAsNumber = _toPrecision.min, nextMaxAsNumber = _toPrecision.max;
            var newNextMin;
            if (!_utils.isFiniteNumber(minBound) && currentRangeMin === nextMinAsNumber) newNextMin = undefined;
            else if (_utils.isFiniteNumber(minBound) && isResetMin) newNextMin = minBound;
            else newNextMin = nextMinAsNumber;
            var newNextMax;
            if (!_utils.isFiniteNumber(maxBound) && currentRangeMax === nextMaxAsNumber) newNextMax = undefined;
            else if (_utils.isFiniteNumber(maxBound) && isResetMax) newNextMax = maxBound;
            else newNextMax = nextMaxAsNumber;
            var isResetNewNextMin = newNextMin === undefined;
            var isGreaterThanCurrentRange = _utils.isFiniteNumber(currentRangeMin) && currentRangeMin <= newNextMin;
            var isMinValid = isResetNewNextMin || _utils.isFiniteNumber(newNextMin) && (!_utils.isFiniteNumber(currentRangeMin) || isGreaterThanCurrentRange);
            var isResetNewNextMax = newNextMax === undefined;
            var isLowerThanRange = _utils.isFiniteNumber(newNextMax) && currentRangeMax >= newNextMax;
            var isMaxValid = isResetNewNextMax || _utils.isFiniteNumber(newNextMax) && (!_utils.isFiniteNumber(currentRangeMax) || isLowerThanRange);
            var hasMinChange = min !== newNextMin;
            var hasMaxChange = max !== newNextMax;
            if ((hasMinChange || hasMaxChange) && isMinValid && isMaxValid) {
                resolvedState = resolvedState.removeNumericRefinement(attribute);
                if (_utils.isFiniteNumber(newNextMin)) resolvedState = resolvedState.addNumericRefinement(attribute, '>=', newNextMin);
                if (_utils.isFiniteNumber(newNextMax)) resolvedState = resolvedState.addNumericRefinement(attribute, '<=', newNextMax);
                return resolvedState;
            }
            return null;
        };
        var sendEventWithRefinedState = function sendEventWithRefinedState(refinedState, instantSearchInstance, helper) {
            var eventName = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : 'Filter Applied';
            var filters = _utils.convertNumericRefinementsToFilters(refinedState, attribute);
            if (filters && filters.length > 0) instantSearchInstance.sendEventToInsights({
                insightsMethod: 'clickedFilters',
                widgetType: $$type,
                eventType: 'click',
                payload: {
                    eventName: eventName,
                    index: helper.getIndex(),
                    filters: filters
                },
                attribute: attribute
            });
        };
        var createSendEvent = function createSendEvent(instantSearchInstance, helper, currentRange) {
            return function() {
                for(var _len = arguments.length, args = new Array(_len), _key = 0; _key < _len; _key++)args[_key] = arguments[_key];
                if (args.length === 1) {
                    instantSearchInstance.sendEventToInsights(args[0]);
                    return;
                }
                var eventType = args[0], facetValue = args[1], eventName = args[2];
                if (eventType !== 'click') return;
                var _facetValue = _slicedToArray(facetValue, 2), nextMin = _facetValue[0], nextMax = _facetValue[1];
                var refinedState = getRefinedState(helper, currentRange, nextMin, nextMax);
                sendEventWithRefinedState(refinedState, instantSearchInstance, helper, eventName);
            };
        };
        function _getCurrentRange(stats) {
            var min;
            if (_utils.isFiniteNumber(minBound)) min = minBound;
            else if (_utils.isFiniteNumber(stats.min)) min = stats.min;
            else min = 0;
            var max;
            if (_utils.isFiniteNumber(maxBound)) max = maxBound;
            else if (_utils.isFiniteNumber(stats.max)) max = stats.max;
            else max = 0;
            return toPrecision({
                min: min,
                max: max,
                precision: precision
            });
        }
        function _getCurrentRefinement(helper) {
            var _ref7 = helper.getNumericRefinement(attribute, '>=') || [], _ref8 = _slicedToArray(_ref7, 1), minValue = _ref8[0];
            var _ref9 = helper.getNumericRefinement(attribute, '<=') || [], _ref10 = _slicedToArray(_ref9, 1), maxValue = _ref10[0];
            var min = _utils.isFiniteNumber(minValue) ? minValue : -Infinity;
            var max = _utils.isFiniteNumber(maxValue) ? maxValue : Infinity;
            return [
                min,
                max
            ];
        }
        function _refine(instantSearchInstance, helper, currentRange) {
            return function() {
                var _ref11 = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : [
                    undefined,
                    undefined
                ], _ref12 = _slicedToArray(_ref11, 2), nextMin = _ref12[0], nextMax = _ref12[1];
                var refinedState = getRefinedState(helper, currentRange, nextMin, nextMax);
                if (refinedState) {
                    sendEventWithRefinedState(refinedState, instantSearchInstance, helper);
                    helper.setState(refinedState).search();
                }
            };
        }
        return {
            $$type: $$type,
            init: function init(initOptions) {
                renderFn(_objectSpread(_objectSpread({}, this.getWidgetRenderState(initOptions)), {}, {
                    instantSearchInstance: initOptions.instantSearchInstance
                }), true);
            },
            render: function render(renderOptions) {
                renderFn(_objectSpread(_objectSpread({}, this.getWidgetRenderState(renderOptions)), {}, {
                    instantSearchInstance: renderOptions.instantSearchInstance
                }), false);
            },
            getRenderState: function getRenderState(renderState, renderOptions) {
                return _objectSpread(_objectSpread({}, renderState), {}, {
                    range: _objectSpread(_objectSpread({}, renderState.range), {}, _defineProperty({}, attribute, this.getWidgetRenderState(renderOptions)))
                });
            },
            getWidgetRenderState: function getWidgetRenderState(_ref13) {
                var results = _ref13.results, helper = _ref13.helper, instantSearchInstance = _ref13.instantSearchInstance;
                var facetsFromResults = results && results.disjunctiveFacets || [];
                var facet = _utils.find(facetsFromResults, function(facetResult) {
                    return facetResult.name === attribute;
                });
                var stats = facet && facet.stats || {
                    min: undefined,
                    max: undefined
                };
                var currentRange = _getCurrentRange(stats);
                var start = _getCurrentRefinement(helper);
                var refine;
                if (!results) // On first render pass an empty range
                // to be able to bypass the validation
                // related to it
                refine = _refine(instantSearchInstance, helper, {
                    min: undefined,
                    max: undefined
                });
                else refine = _refine(instantSearchInstance, helper, currentRange);
                return {
                    refine: refine,
                    canRefine: currentRange.min !== currentRange.max,
                    format: rangeFormatter,
                    range: currentRange,
                    sendEvent: createSendEvent(instantSearchInstance, helper, currentRange),
                    widgetParams: _objectSpread(_objectSpread({}, widgetParams), {}, {
                        precision: precision
                    }),
                    start: start
                };
            },
            dispose: function dispose(_ref14) {
                var state = _ref14.state;
                unmountFn();
                return state.removeDisjunctiveFacet(attribute).removeNumericRefinement(attribute);
            },
            getWidgetUiState: function getWidgetUiState(uiState, _ref15) {
                var searchParameters = _ref15.searchParameters;
                var _searchParameters$get = searchParameters.getNumericRefinements(attribute), _searchParameters$get2 = _searchParameters$get['>='], min = _searchParameters$get2 === void 0 ? [] : _searchParameters$get2, _searchParameters$get3 = _searchParameters$get['<='], max = _searchParameters$get3 === void 0 ? [] : _searchParameters$get3;
                if (min.length === 0 && max.length === 0) return uiState;
                return _objectSpread(_objectSpread({}, uiState), {}, {
                    range: _objectSpread(_objectSpread({}, uiState.range), {}, _defineProperty({}, attribute, "".concat(min, ":").concat(max)))
                });
            },
            getWidgetSearchParameters: function getWidgetSearchParameters(searchParameters, _ref16) {
                var uiState = _ref16.uiState;
                var widgetSearchParameters = searchParameters.addDisjunctiveFacet(attribute).setQueryParameters({
                    numericRefinements: _objectSpread(_objectSpread({}, searchParameters.numericRefinements), {}, _defineProperty({}, attribute, {}))
                });
                if (_utils.isFiniteNumber(minBound)) widgetSearchParameters = widgetSearchParameters.addNumericRefinement(attribute, '>=', minBound);
                if (_utils.isFiniteNumber(maxBound)) widgetSearchParameters = widgetSearchParameters.addNumericRefinement(attribute, '<=', maxBound);
                var value = uiState.range && uiState.range[attribute];
                if (!value || value.indexOf(':') === -1) return widgetSearchParameters;
                var _value$split$map = value.split(':').map(parseFloat), _value$split$map2 = _slicedToArray(_value$split$map, 2), lowerBound = _value$split$map2[0], upperBound = _value$split$map2[1];
                if (_utils.isFiniteNumber(lowerBound) && (!_utils.isFiniteNumber(minBound) || minBound < lowerBound)) {
                    widgetSearchParameters = widgetSearchParameters.removeNumericRefinement(attribute, '>=');
                    widgetSearchParameters = widgetSearchParameters.addNumericRefinement(attribute, '>=', lowerBound);
                }
                if (_utils.isFiniteNumber(upperBound) && (!_utils.isFiniteNumber(maxBound) || upperBound < maxBound)) {
                    widgetSearchParameters = widgetSearchParameters.removeNumericRefinement(attribute, '<=');
                    widgetSearchParameters = widgetSearchParameters.addNumericRefinement(attribute, '<=', upperBound);
                }
                return widgetSearchParameters;
            }
        };
    };
};
exports.default = connectRange;

},{"../../lib/utils":"etVYs","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"kkKYv":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _utils = require("../../lib/utils");
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        if (enumerableOnly) symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        });
        keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = arguments[i] != null ? arguments[i] : {};
        if (i % 2) ownKeys(Object(source), true).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        });
        else if (Object.getOwnPropertyDescriptors) Object.defineProperties(target, Object.getOwnPropertyDescriptors(source));
        else ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
function _objectWithoutProperties(source, excluded) {
    if (source == null) return {};
    var target = _objectWithoutPropertiesLoose(source, excluded);
    var key, i;
    if (Object.getOwnPropertySymbols) {
        var sourceSymbolKeys = Object.getOwnPropertySymbols(source);
        for(i = 0; i < sourceSymbolKeys.length; i++){
            key = sourceSymbolKeys[i];
            if (excluded.indexOf(key) >= 0) continue;
            if (!Object.prototype.propertyIsEnumerable.call(source, key)) continue;
            target[key] = source[key];
        }
    }
    return target;
}
function _objectWithoutPropertiesLoose(source, excluded) {
    if (source == null) return {};
    var target = {};
    var sourceKeys = Object.keys(source);
    var key, i;
    for(i = 0; i < sourceKeys.length; i++){
        key = sourceKeys[i];
        if (excluded.indexOf(key) >= 0) continue;
        target[key] = source[key];
    }
    return target;
}
var withUsage = _utils.createDocumentationMessageGenerator({
    name: 'refinement-list',
    connector: true
});
/**
 * **RefinementList** connector provides the logic to build a custom widget that
 * will let the user filter the results based on the values of a specific facet.
 *
 * **Requirement:** the attribute passed as `attribute` must be present in
 * attributesForFaceting of the searched index.
 *
 * This connector provides:
 * - a `refine()` function to select an item.
 * - a `toggleShowMore()` function to display more or less items
 * - a `searchForItems()` function to search within the items.
 */ var connectRefinementList = function connectRefinementList(renderFn) {
    var unmountFn = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : _utils.noop;
    _utils.checkRendering(renderFn, withUsage());
    return function(widgetParams) {
        var _ref = widgetParams || {}, attribute = _ref.attribute, _ref$operator = _ref.operator, operator = _ref$operator === void 0 ? 'or' : _ref$operator, _ref$limit = _ref.limit, limit = _ref$limit === void 0 ? 10 : _ref$limit, _ref$showMore = _ref.showMore, showMore = _ref$showMore === void 0 ? false : _ref$showMore, _ref$showMoreLimit = _ref.showMoreLimit, showMoreLimit = _ref$showMoreLimit === void 0 ? 20 : _ref$showMoreLimit, _ref$sortBy = _ref.sortBy, sortBy = _ref$sortBy === void 0 ? [
            'isRefined',
            'count:desc',
            'name:asc'
        ] : _ref$sortBy, _ref$escapeFacetValue = _ref.escapeFacetValues, escapeFacetValues = _ref$escapeFacetValue === void 0 ? true : _ref$escapeFacetValue, _ref$transformItems = _ref.transformItems, transformItems = _ref$transformItems === void 0 ? function(items) {
            return items;
        } : _ref$transformItems;
        if (!attribute) throw new Error(withUsage('The `attribute` option is required.'));
        if (!/^(and|or)$/.test(operator)) throw new Error(withUsage("The `operator` must one of: `\"and\"`, `\"or\"` (got \"".concat(operator, "\").")));
        if (showMore === true && showMoreLimit <= limit) throw new Error(withUsage('`showMoreLimit` should be greater than `limit`.'));
        var formatItems = function formatItems(_ref2) {
            var label = _ref2.name, item = _objectWithoutProperties(_ref2, [
                "name"
            ]);
            return _objectSpread(_objectSpread({}, item), {}, {
                label: label,
                value: label,
                highlighted: label
            });
        };
        var lastResultsFromMainSearch;
        var lastItemsFromMainSearch = [];
        var hasExhaustiveItems = true;
        var triggerRefine;
        var sendEvent;
        var isShowingMore = false; // Provide the same function to the `renderFn` so that way the user
        // has to only bind it once when `isFirstRendering` for instance
        var toggleShowMore = function toggleShowMore() {};
        function cachedToggleShowMore() {
            toggleShowMore();
        }
        function createToggleShowMore(renderOptions, widget) {
            return function() {
                isShowingMore = !isShowingMore;
                widget.render(renderOptions);
            };
        }
        function getLimit() {
            return isShowingMore ? showMoreLimit : limit;
        }
        var searchForFacetValues = function searchForFacetValues() {
            return function() {};
        };
        var createSearchForFacetValues = function createSearchForFacetValues(helper, widget) {
            return function(renderOptions) {
                return function(query) {
                    var instantSearchInstance = renderOptions.instantSearchInstance;
                    if (query === '' && lastItemsFromMainSearch) // render with previous data from the helper.
                    renderFn(_objectSpread(_objectSpread({}, widget.getWidgetRenderState(_objectSpread(_objectSpread({}, renderOptions), {}, {
                        results: lastResultsFromMainSearch
                    }))), {}, {
                        instantSearchInstance: instantSearchInstance
                    }), false);
                    else {
                        var tags = {
                            highlightPreTag: escapeFacetValues ? _utils.TAG_PLACEHOLDER.highlightPreTag : _utils.TAG_REPLACEMENT.highlightPreTag,
                            highlightPostTag: escapeFacetValues ? _utils.TAG_PLACEHOLDER.highlightPostTag : _utils.TAG_REPLACEMENT.highlightPostTag
                        };
                        helper.searchForFacetValues(attribute, query, // doesn't support a greater number.
                        // See https://www.algolia.com/doc/api-reference/api-parameters/maxFacetHits/
                        Math.min(getLimit(), 100), tags).then(function(results) {
                            var facetValues = escapeFacetValues ? _utils.escapeFacets(results.facetHits) : results.facetHits;
                            var normalizedFacetValues = transformItems(facetValues.map(function(_ref3) {
                                var value = _ref3.value, item = _objectWithoutProperties(_ref3, [
                                    "value"
                                ]);
                                return _objectSpread(_objectSpread({}, item), {}, {
                                    value: value,
                                    label: value
                                });
                            }));
                            renderFn(_objectSpread(_objectSpread({}, widget.getWidgetRenderState(_objectSpread(_objectSpread({}, renderOptions), {}, {
                                results: lastResultsFromMainSearch
                            }))), {}, {
                                items: normalizedFacetValues,
                                canToggleShowMore: false,
                                canRefine: true,
                                isFromSearch: true,
                                instantSearchInstance: instantSearchInstance
                            }), false);
                        });
                    }
                };
            };
        };
        return {
            $$type: 'ais.refinementList',
            init: function init(initOptions) {
                renderFn(_objectSpread(_objectSpread({}, this.getWidgetRenderState(initOptions)), {}, {
                    instantSearchInstance: initOptions.instantSearchInstance
                }), true);
            },
            render: function render(renderOptions) {
                renderFn(_objectSpread(_objectSpread({}, this.getWidgetRenderState(renderOptions)), {}, {
                    instantSearchInstance: renderOptions.instantSearchInstance
                }), false);
            },
            getRenderState: function getRenderState(renderState, renderOptions) {
                return _objectSpread(_objectSpread({}, renderState), {}, {
                    refinementList: _objectSpread(_objectSpread({}, renderState.refinementList), {}, _defineProperty({}, attribute, this.getWidgetRenderState(renderOptions)))
                });
            },
            getWidgetRenderState: function getWidgetRenderState(renderOptions) {
                var results = renderOptions.results, state = renderOptions.state, _createURL = renderOptions.createURL, instantSearchInstance = renderOptions.instantSearchInstance, helper = renderOptions.helper;
                var items = [];
                var facetValues = [];
                if (!sendEvent || !triggerRefine || !searchForFacetValues) {
                    sendEvent = _utils.createSendEventForFacet({
                        instantSearchInstance: instantSearchInstance,
                        helper: helper,
                        attribute: attribute,
                        widgetType: this.$$type
                    });
                    triggerRefine = function triggerRefine(facetValue) {
                        sendEvent('click', facetValue);
                        helper.toggleFacetRefinement(attribute, facetValue).search();
                    };
                    searchForFacetValues = createSearchForFacetValues(helper, this);
                }
                if (results) {
                    var values = results.getFacetValues(attribute, {
                        sortBy: sortBy
                    });
                    facetValues = values && Array.isArray(values) ? values : [];
                    items = transformItems(facetValues.slice(0, getLimit()).map(formatItems));
                    var maxValuesPerFacetConfig = state.maxValuesPerFacet;
                    var currentLimit = getLimit(); // If the limit is the max number of facet retrieved it is impossible to know
                    // if the facets are exhaustive. The only moment we are sure it is exhaustive
                    // is when it is strictly under the number requested unless we know that another
                    // widget has requested more values (maxValuesPerFacet > getLimit()).
                    // Because this is used for making the search of facets unable or not, it is important
                    // to be conservative here.
                    hasExhaustiveItems = maxValuesPerFacetConfig > currentLimit ? facetValues.length <= currentLimit : facetValues.length < currentLimit;
                    lastResultsFromMainSearch = results;
                    lastItemsFromMainSearch = items;
                    if (renderOptions.results) toggleShowMore = createToggleShowMore(renderOptions, this);
                } // Do not mistake searchForFacetValues and searchFacetValues which is the actual search
                // function
                var searchFacetValues = searchForFacetValues && searchForFacetValues(renderOptions);
                var canShowLess = isShowingMore && lastItemsFromMainSearch.length > limit;
                var canShowMore = showMore && !hasExhaustiveItems;
                var canToggleShowMore = canShowLess || canShowMore;
                return {
                    createURL: function createURL(facetValue) {
                        return _createURL(state.resetPage().toggleFacetRefinement(attribute, facetValue));
                    },
                    items: items,
                    refine: triggerRefine,
                    searchForItems: searchFacetValues,
                    isFromSearch: false,
                    canRefine: items.length > 0,
                    widgetParams: widgetParams,
                    isShowingMore: isShowingMore,
                    canToggleShowMore: canToggleShowMore,
                    toggleShowMore: cachedToggleShowMore,
                    sendEvent: sendEvent,
                    hasExhaustiveItems: hasExhaustiveItems
                };
            },
            dispose: function dispose(_ref4) {
                var state = _ref4.state;
                unmountFn();
                var withoutMaxValuesPerFacet = state.setQueryParameter('maxValuesPerFacet', undefined);
                if (operator === 'and') return withoutMaxValuesPerFacet.removeFacet(attribute);
                return withoutMaxValuesPerFacet.removeDisjunctiveFacet(attribute);
            },
            getWidgetUiState: function getWidgetUiState(uiState, _ref5) {
                var searchParameters = _ref5.searchParameters;
                var values = operator === 'or' ? searchParameters.getDisjunctiveRefinements(attribute) : searchParameters.getConjunctiveRefinements(attribute);
                if (!values.length) return uiState;
                return _objectSpread(_objectSpread({}, uiState), {}, {
                    refinementList: _objectSpread(_objectSpread({}, uiState.refinementList), {}, _defineProperty({}, attribute, values))
                });
            },
            getWidgetSearchParameters: function getWidgetSearchParameters(searchParameters, _ref6) {
                var uiState = _ref6.uiState;
                var isDisjunctive = operator === 'or';
                var values = uiState.refinementList && uiState.refinementList[attribute];
                var withoutRefinements = searchParameters.clearRefinements(attribute);
                var withFacetConfiguration = isDisjunctive ? withoutRefinements.addDisjunctiveFacet(attribute) : withoutRefinements.addFacet(attribute);
                var currentMaxValuesPerFacet = withFacetConfiguration.maxValuesPerFacet || 0;
                var nextMaxValuesPerFacet = Math.max(currentMaxValuesPerFacet, showMore ? showMoreLimit : limit);
                var withMaxValuesPerFacet = withFacetConfiguration.setQueryParameter('maxValuesPerFacet', nextMaxValuesPerFacet);
                if (!values) {
                    var key = isDisjunctive ? 'disjunctiveFacetsRefinements' : 'facetsRefinements';
                    return withMaxValuesPerFacet.setQueryParameters(_defineProperty({}, key, _objectSpread(_objectSpread({}, withMaxValuesPerFacet[key]), {}, _defineProperty({}, attribute, []))));
                }
                return values.reduce(function(parameters, value) {
                    return isDisjunctive ? parameters.addDisjunctiveFacetRefinement(attribute, value) : parameters.addFacetRefinement(attribute, value);
                }, withMaxValuesPerFacet);
            }
        };
    };
};
exports.default = connectRefinementList;

},{"../../lib/utils":"etVYs","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"kqCmi":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _utils = require("../../lib/utils");
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        if (enumerableOnly) symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        });
        keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = arguments[i] != null ? arguments[i] : {};
        if (i % 2) ownKeys(Object(source), true).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        });
        else if (Object.getOwnPropertyDescriptors) Object.defineProperties(target, Object.getOwnPropertyDescriptors(source));
        else ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
var withUsage = _utils.createDocumentationMessageGenerator({
    name: 'search-box',
    connector: true
});
/**
 * **SearchBox** connector provides the logic to build a widget that will let the user search for a query.
 *
 * The connector provides to the rendering: `refine()` to set the query. The behaviour of this function
 * may be impacted by the `queryHook` widget parameter.
 */ var connectSearchBox = function connectSearchBox(renderFn) {
    var unmountFn = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : _utils.noop;
    _utils.checkRendering(renderFn, withUsage());
    return function(widgetParams) {
        var _ref = widgetParams || {}, queryHook = _ref.queryHook;
        function clear(helper) {
            return function() {
                helper.setQuery('').search();
            };
        }
        var _refine;
        var _clear = function _clear() {};
        function _cachedClear() {
            _clear();
        }
        return {
            $$type: 'ais.searchBox',
            init: function init(initOptions) {
                var instantSearchInstance = initOptions.instantSearchInstance;
                renderFn(_objectSpread(_objectSpread({}, this.getWidgetRenderState(initOptions)), {}, {
                    instantSearchInstance: instantSearchInstance
                }), true);
            },
            render: function render(renderOptions) {
                var instantSearchInstance = renderOptions.instantSearchInstance;
                renderFn(_objectSpread(_objectSpread({}, this.getWidgetRenderState(renderOptions)), {}, {
                    instantSearchInstance: instantSearchInstance
                }), false);
            },
            dispose: function dispose(_ref2) {
                var state = _ref2.state;
                unmountFn();
                return state.setQueryParameter('query', undefined);
            },
            getRenderState: function getRenderState(renderState, renderOptions) {
                return _objectSpread(_objectSpread({}, renderState), {}, {
                    searchBox: this.getWidgetRenderState(renderOptions)
                });
            },
            getWidgetRenderState: function getWidgetRenderState(_ref3) {
                var helper = _ref3.helper, searchMetadata = _ref3.searchMetadata;
                if (!_refine) {
                    var setQueryAndSearch = function setQueryAndSearch(query) {
                        if (query !== helper.state.query) helper.setQuery(query).search();
                    };
                    _refine = function _refine(query) {
                        if (queryHook) {
                            queryHook(query, setQueryAndSearch);
                            return;
                        }
                        setQueryAndSearch(query);
                    };
                }
                _clear = clear(helper);
                return {
                    query: helper.state.query || '',
                    refine: _refine,
                    clear: _cachedClear,
                    widgetParams: widgetParams,
                    isSearchStalled: searchMetadata.isSearchStalled
                };
            },
            getWidgetUiState: function getWidgetUiState(uiState, _ref4) {
                var searchParameters = _ref4.searchParameters;
                var query = searchParameters.query || '';
                if (query === '' || uiState && uiState.query === query) return uiState;
                return _objectSpread(_objectSpread({}, uiState), {}, {
                    query: query
                });
            },
            getWidgetSearchParameters: function getWidgetSearchParameters(searchParameters, _ref5) {
                var uiState = _ref5.uiState;
                return searchParameters.setQueryParameter('query', uiState.query || '');
            }
        };
    };
};
exports.default = connectSearchBox;

},{"../../lib/utils":"etVYs","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"3pFgJ":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _utils = require("../../lib/utils");
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        if (enumerableOnly) symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        });
        keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = arguments[i] != null ? arguments[i] : {};
        if (i % 2) ownKeys(Object(source), true).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        });
        else if (Object.getOwnPropertyDescriptors) Object.defineProperties(target, Object.getOwnPropertyDescriptors(source));
        else ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
var withUsage = _utils.createDocumentationMessageGenerator({
    name: 'sort-by',
    connector: true
});
/**
 * The **SortBy** connector provides the logic to build a custom widget that will display a
 * list of indices. With Algolia, this is most commonly used for changing ranking strategy. This allows
 * a user to change how the hits are being sorted.
 */ var connectSortBy = function connectSortBy(renderFn) {
    var unmountFn = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : _utils.noop;
    _utils.checkRendering(renderFn, withUsage());
    var connectorState = {};
    return function(widgetParams) {
        var _ref = widgetParams || {}, items = _ref.items, _ref$transformItems = _ref.transformItems, transformItems = _ref$transformItems === void 0 ? function(x) {
            return x;
        } : _ref$transformItems;
        if (!Array.isArray(items)) throw new Error(withUsage('The `items` option expects an array of objects.'));
        return {
            $$type: 'ais.sortBy',
            init: function init(initOptions) {
                var instantSearchInstance = initOptions.instantSearchInstance;
                var widgetRenderState = this.getWidgetRenderState(initOptions);
                var currentIndex = widgetRenderState.currentRefinement;
                var isCurrentIndexInItems = _utils.find(items, function(item) {
                    return item.value === currentIndex;
                });
                _utils.warning(isCurrentIndexInItems !== undefined, "The index named \"".concat(currentIndex, "\" is not listed in the `items` of `sortBy`."));
                renderFn(_objectSpread(_objectSpread({}, widgetRenderState), {}, {
                    instantSearchInstance: instantSearchInstance
                }), true);
            },
            render: function render(renderOptions) {
                var instantSearchInstance = renderOptions.instantSearchInstance;
                renderFn(_objectSpread(_objectSpread({}, this.getWidgetRenderState(renderOptions)), {}, {
                    instantSearchInstance: instantSearchInstance
                }), false);
            },
            dispose: function dispose(_ref2) {
                var state = _ref2.state;
                unmountFn();
                return connectorState.initialIndex ? state.setIndex(connectorState.initialIndex) : state;
            },
            getRenderState: function getRenderState(renderState, renderOptions) {
                return _objectSpread(_objectSpread({}, renderState), {}, {
                    sortBy: this.getWidgetRenderState(renderOptions)
                });
            },
            getWidgetRenderState: function getWidgetRenderState(_ref3) {
                var results = _ref3.results, helper = _ref3.helper, parent = _ref3.parent;
                if (!connectorState.initialIndex && parent) connectorState.initialIndex = parent.getIndexName();
                if (!connectorState.setIndex) connectorState.setIndex = function(indexName) {
                    helper.setIndex(indexName).search();
                };
                return {
                    currentRefinement: helper.state.index,
                    options: transformItems(items),
                    refine: connectorState.setIndex,
                    hasNoResults: results ? results.nbHits === 0 : true,
                    widgetParams: widgetParams
                };
            },
            getWidgetUiState: function getWidgetUiState(uiState, _ref4) {
                var searchParameters = _ref4.searchParameters;
                var currentIndex = searchParameters.index;
                if (currentIndex === connectorState.initialIndex) return uiState;
                return _objectSpread(_objectSpread({}, uiState), {}, {
                    sortBy: currentIndex
                });
            },
            getWidgetSearchParameters: function getWidgetSearchParameters(searchParameters, _ref5) {
                var uiState = _ref5.uiState;
                return searchParameters.setQueryParameter('index', uiState.sortBy || connectorState.initialIndex || searchParameters.index);
            }
        };
    };
};
exports.default = connectSortBy;

},{"../../lib/utils":"etVYs","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"lvgHS":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _algoliasearchHelper = require("algoliasearch-helper");
var _algoliasearchHelperDefault = parcelHelpers.interopDefault(_algoliasearchHelper);
var _utils = require("../../lib/utils");
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        if (enumerableOnly) symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        });
        keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = arguments[i] != null ? arguments[i] : {};
        if (i % 2) ownKeys(Object(source), true).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        });
        else if (Object.getOwnPropertyDescriptors) Object.defineProperties(target, Object.getOwnPropertyDescriptors(source));
        else ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
/**
 * Refine the given search parameters.
 */ var withUsage = _utils.createDocumentationMessageGenerator({
    name: 'configure',
    connector: true
});
function getInitialSearchParameters(state, widgetParams) {
    // We leverage the helper internals to remove the `widgetParams` from
    // the state. The function `setQueryParameters` omits the values that
    // are `undefined` on the next state.
    return state.setQueryParameters(Object.keys(widgetParams.searchParameters).reduce(function(acc, key) {
        return _objectSpread(_objectSpread({}, acc), {}, _defineProperty({}, key, undefined));
    }, {}));
}
var connectConfigure = function connectConfigure() {
    var renderFn = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : _utils.noop;
    var unmountFn = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : _utils.noop;
    return function(widgetParams) {
        if (!widgetParams || !_utils.isPlainObject(widgetParams.searchParameters)) throw new Error(withUsage('The `searchParameters` option expects an object.'));
        var connectorState = {};
        function refine(helper) {
            return function(searchParameters) {
                // Merge new `searchParameters` with the ones set from other widgets
                var actualState = getInitialSearchParameters(helper.state, widgetParams);
                var nextSearchParameters = _utils.mergeSearchParameters(actualState, new _algoliasearchHelperDefault.default.SearchParameters(searchParameters)); // Update original `widgetParams.searchParameters` to the new refined one
                widgetParams.searchParameters = searchParameters; // Trigger a search with the resolved search parameters
                helper.setState(nextSearchParameters).search();
            };
        }
        return {
            $$type: 'ais.configure',
            init: function init(initOptions) {
                var instantSearchInstance = initOptions.instantSearchInstance;
                renderFn(_objectSpread(_objectSpread({}, this.getWidgetRenderState(initOptions)), {}, {
                    instantSearchInstance: instantSearchInstance
                }), true);
            },
            render: function render(renderOptions) {
                var instantSearchInstance = renderOptions.instantSearchInstance;
                renderFn(_objectSpread(_objectSpread({}, this.getWidgetRenderState(renderOptions)), {}, {
                    instantSearchInstance: instantSearchInstance
                }), false);
            },
            dispose: function dispose(_ref) {
                var state = _ref.state;
                unmountFn();
                return getInitialSearchParameters(state, widgetParams);
            },
            getRenderState: function getRenderState(renderState, renderOptions) {
                var _renderState$configur;
                var widgetRenderState = this.getWidgetRenderState(renderOptions);
                return _objectSpread(_objectSpread({}, renderState), {}, {
                    configure: _objectSpread(_objectSpread({}, widgetRenderState), {}, {
                        widgetParams: _objectSpread(_objectSpread({}, widgetRenderState.widgetParams), {}, {
                            searchParameters: _utils.mergeSearchParameters(new _algoliasearchHelperDefault.default.SearchParameters((_renderState$configur = renderState.configure) === null || _renderState$configur === void 0 ? void 0 : _renderState$configur.widgetParams.searchParameters), new _algoliasearchHelperDefault.default.SearchParameters(widgetRenderState.widgetParams.searchParameters)).getQueryParams()
                        })
                    })
                });
            },
            getWidgetRenderState: function getWidgetRenderState(_ref2) {
                var helper = _ref2.helper;
                if (!connectorState.refine) connectorState.refine = refine(helper);
                return {
                    refine: connectorState.refine,
                    widgetParams: widgetParams
                };
            },
            getWidgetSearchParameters: function getWidgetSearchParameters(state, _ref3) {
                var uiState = _ref3.uiState;
                return _utils.mergeSearchParameters(state, new _algoliasearchHelperDefault.default.SearchParameters(_objectSpread(_objectSpread({}, uiState.configure), widgetParams.searchParameters)));
            },
            getWidgetUiState: function getWidgetUiState(uiState) {
                return _objectSpread(_objectSpread({}, uiState), {}, {
                    configure: _objectSpread(_objectSpread({}, uiState.configure), widgetParams.searchParameters)
                });
            }
        };
    };
};
exports.default = connectConfigure;

},{"algoliasearch-helper":"jGqjt","../../lib/utils":"etVYs","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"kDyli":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _createQuerySuggestionsPlugin = require("./createQuerySuggestionsPlugin");
parcelHelpers.exportAll(_createQuerySuggestionsPlugin, exports);
var _getTemplates = require("./getTemplates");
parcelHelpers.exportAll(_getTemplates, exports);

},{"./createQuerySuggestionsPlugin":"7BBqc","./getTemplates":"9IWzT","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"7BBqc":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "createQuerySuggestionsPlugin", ()=>createQuerySuggestionsPlugin
);
var _autocompleteJs = require("@algolia/autocomplete-js");
var _autocompleteShared = require("@algolia/autocomplete-shared");
var _getTemplates = require("./getTemplates");
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        enumerableOnly && (symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        })), keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = null != arguments[i] ? arguments[i] : {};
        i % 2 ? ownKeys(Object(source), !0).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
function _createForOfIteratorHelper(o, allowArrayLike) {
    var it = typeof Symbol !== "undefined" && o[Symbol.iterator] || o["@@iterator"];
    if (!it) {
        if (Array.isArray(o) || (it = _unsupportedIterableToArray(o)) || allowArrayLike && o && typeof o.length === "number") {
            if (it) o = it;
            var i = 0;
            var F = function F() {};
            return {
                s: F,
                n: function n() {
                    if (i >= o.length) return {
                        done: true
                    };
                    return {
                        done: false,
                        value: o[i++]
                    };
                },
                e: function e(_e) {
                    throw _e;
                },
                f: F
            };
        }
        throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
    }
    var normalCompletion = true, didErr = false, err;
    return {
        s: function s() {
            it = it.call(o);
        },
        n: function n() {
            var step = it.next();
            normalCompletion = step.done;
            return step;
        },
        e: function e(_e2) {
            didErr = true;
            err = _e2;
        },
        f: function f() {
            try {
                if (!normalCompletion && it.return != null) it.return();
            } finally{
                if (didErr) throw err;
            }
        }
    };
}
function _unsupportedIterableToArray(o, minLen) {
    if (!o) return;
    if (typeof o === "string") return _arrayLikeToArray(o, minLen);
    var n = Object.prototype.toString.call(o).slice(8, -1);
    if (n === "Object" && o.constructor) n = o.constructor.name;
    if (n === "Map" || n === "Set") return Array.from(o);
    if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen);
}
function _arrayLikeToArray(arr, len) {
    if (len == null || len > arr.length) len = arr.length;
    for(var i = 0, arr2 = new Array(len); i < len; i++)arr2[i] = arr[i];
    return arr2;
}
function createQuerySuggestionsPlugin(options) {
    var _getOptions = getOptions(options), searchClient = _getOptions.searchClient, indexName = _getOptions.indexName, getSearchParams = _getOptions.getSearchParams, transformSource = _getOptions.transformSource, categoryAttribute = _getOptions.categoryAttribute, itemsWithCategories = _getOptions.itemsWithCategories, categoriesPerItem = _getOptions.categoriesPerItem;
    return {
        name: 'aa.querySuggestionsPlugin',
        getSources: function getSources(_ref) {
            var query = _ref.query, setQuery = _ref.setQuery, refresh = _ref.refresh, state = _ref.state;
            function onTapAhead(item) {
                setQuery("".concat(item.query, " "));
                refresh();
            }
            return [
                transformSource({
                    source: {
                        sourceId: 'querySuggestionsPlugin',
                        getItemInputValue: function getItemInputValue(_ref2) {
                            var item = _ref2.item;
                            return item.query;
                        },
                        getItems: function getItems() {
                            return _autocompleteJs.getAlgoliaResults({
                                searchClient: searchClient,
                                queries: [
                                    {
                                        indexName: indexName,
                                        query: query,
                                        params: getSearchParams({
                                            state: state
                                        })
                                    }
                                ],
                                transformResponse: function transformResponse(_ref3) {
                                    var hits = _ref3.hits;
                                    var querySuggestionsHits = hits[0];
                                    if (!query || !categoryAttribute) return querySuggestionsHits;
                                    return querySuggestionsHits.reduce(function(acc, current, i) {
                                        var items = [
                                            current
                                        ];
                                        if (i <= itemsWithCategories - 1) {
                                            var categories = _autocompleteShared.getAttributeValueByPath(current, Array.isArray(categoryAttribute) ? categoryAttribute : [
                                                categoryAttribute
                                            ]).map(function(x) {
                                                return x.value;
                                            }).slice(0, categoriesPerItem);
                                            var _iterator = _createForOfIteratorHelper(categories), _step;
                                            try {
                                                for(_iterator.s(); !(_step = _iterator.n()).done;){
                                                    var category = _step.value;
                                                    items.push(_objectSpread({
                                                        __autocomplete_qsCategory: category
                                                    }, current));
                                                }
                                            } catch (err) {
                                                _iterator.e(err);
                                            } finally{
                                                _iterator.f();
                                            }
                                        }
                                        acc.push.apply(acc, items);
                                        return acc;
                                    }, []);
                                }
                            });
                        },
                        templates: _getTemplates.getTemplates({
                            onTapAhead: onTapAhead
                        })
                    },
                    onTapAhead: onTapAhead,
                    state: state
                })
            ];
        },
        __autocomplete_pluginOptions: options
    };
}
function getOptions(options) {
    return _objectSpread({
        getSearchParams: function getSearchParams() {
            return {};
        },
        transformSource: function transformSource(_ref4) {
            var source = _ref4.source;
            return source;
        },
        itemsWithCategories: 1,
        categoriesPerItem: 1
    }, options);
}

},{"@algolia/autocomplete-js":"3Syxs","@algolia/autocomplete-shared":"59T59","./getTemplates":"9IWzT","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"9IWzT":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/** @jsx createElement */ parcelHelpers.export(exports, "getTemplates", ()=>getTemplates
);
function getTemplates(_ref) {
    var onTapAhead = _ref.onTapAhead;
    return {
        item: function item(_ref2) {
            var item = _ref2.item, createElement = _ref2.createElement, components = _ref2.components;
            if (item.__autocomplete_qsCategory) return createElement("div", {
                className: "aa-ItemWrapper"
            }, createElement("div", {
                className: "aa-ItemContent aa-ItemContent--indented"
            }, createElement("div", {
                className: "aa-ItemContentSubtitle aa-ItemContentSubtitle--standalone"
            }, createElement("span", {
                className: "aa-ItemContentSubtitleIcon"
            }), createElement("span", null, "in", ' ', createElement("span", {
                className: "aa-ItemContentSubtitleCategory"
            }, item.__autocomplete_qsCategory)))));
            return createElement("div", {
                className: "aa-ItemWrapper"
            }, createElement("div", {
                className: "aa-ItemContent"
            }, createElement("div", {
                className: "aa-ItemIcon aa-ItemIcon--noBorder"
            }, createElement("svg", {
                viewBox: "0 0 24 24",
                fill: "currentColor"
            }, createElement("path", {
                d: "M16.041 15.856c-0.034 0.026-0.067 0.055-0.099 0.087s-0.060 0.064-0.087 0.099c-1.258 1.213-2.969 1.958-4.855 1.958-1.933 0-3.682-0.782-4.95-2.050s-2.050-3.017-2.050-4.95 0.782-3.682 2.050-4.95 3.017-2.050 4.95-2.050 3.682 0.782 4.95 2.050 2.050 3.017 2.050 4.95c0 1.886-0.745 3.597-1.959 4.856zM21.707 20.293l-3.675-3.675c1.231-1.54 1.968-3.493 1.968-5.618 0-2.485-1.008-4.736-2.636-6.364s-3.879-2.636-6.364-2.636-4.736 1.008-6.364 2.636-2.636 3.879-2.636 6.364 1.008 4.736 2.636 6.364 3.879 2.636 6.364 2.636c2.125 0 4.078-0.737 5.618-1.968l3.675 3.675c0.391 0.391 1.024 0.391 1.414 0s0.391-1.024 0-1.414z"
            }))), createElement("div", {
                className: "aa-ItemContentBody"
            }, createElement("div", {
                className: "aa-ItemContentTitle"
            }, createElement(components.ReverseHighlight, {
                hit: item,
                attribute: "query"
            })))), createElement("div", {
                className: "aa-ItemActions"
            }, createElement("button", {
                className: "aa-ItemActionButton",
                title: "Fill query with \"".concat(item.query, "\""),
                onClick: function onClick(event) {
                    event.preventDefault();
                    event.stopPropagation();
                    onTapAhead(item);
                }
            }, createElement("svg", {
                viewBox: "0 0 24 24",
                fill: "currentColor"
            }, createElement("path", {
                d: "M8 17v-7.586l8.293 8.293c0.391 0.391 1.024 0.391 1.414 0s0.391-1.024 0-1.414l-8.293-8.293h7.586c0.552 0 1-0.448 1-1s-0.448-1-1-1h-10c-0.552 0-1 0.448-1 1v10c0 0.552 0.448 1 1 1s1-0.448 1-1z"
            })))));
        }
    };
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"lFtzN":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _addHighlightedAttribute = require("./addHighlightedAttribute");
parcelHelpers.exportAll(_addHighlightedAttribute, exports);
var _createLocalStorageRecentSearchesPlugin = require("./createLocalStorageRecentSearchesPlugin");
parcelHelpers.exportAll(_createLocalStorageRecentSearchesPlugin, exports);
var _createRecentSearchesPlugin = require("./createRecentSearchesPlugin");
parcelHelpers.exportAll(_createRecentSearchesPlugin, exports);
var _getTemplates = require("./getTemplates");
parcelHelpers.exportAll(_getTemplates, exports);
var _search = require("./search");
parcelHelpers.exportAll(_search, exports);

},{"./addHighlightedAttribute":"iMgYs","./createLocalStorageRecentSearchesPlugin":"gT3kG","./createRecentSearchesPlugin":"2L1S7","./getTemplates":"8XaSk","./search":"CzUZD","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"iMgYs":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "addHighlightedAttribute", ()=>addHighlightedAttribute
);
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        enumerableOnly && (symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        })), keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = null != arguments[i] ? arguments[i] : {};
        i % 2 ? ownKeys(Object(source), !0).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
function addHighlightedAttribute(_ref) {
    var item = _ref.item, query = _ref.query;
    return _objectSpread(_objectSpread({}, item), {}, {
        _highlightResult: {
            label: {
                value: query ? item.label.replace(new RegExp(query.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&'), 'gi'), function(match) {
                    return "__aa-highlight__".concat(match, "__/aa-highlight__");
                }) : item.label
            }
        }
    });
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"gT3kG":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "createLocalStorageRecentSearchesPlugin", ()=>createLocalStorageRecentSearchesPlugin
);
var _constants = require("./constants");
var _createLocalStorage = require("./createLocalStorage");
var _createRecentSearchesPlugin = require("./createRecentSearchesPlugin");
var _search = require("./search");
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        enumerableOnly && (symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        })), keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = null != arguments[i] ? arguments[i] : {};
        i % 2 ? ownKeys(Object(source), !0).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
function createLocalStorageRecentSearchesPlugin(options) {
    var _getOptions = getOptions(options), key = _getOptions.key, limit = _getOptions.limit, transformSource = _getOptions.transformSource, search = _getOptions.search, subscribe = _getOptions.subscribe;
    var storage = _createLocalStorage.createLocalStorage({
        key: [
            _constants.LOCAL_STORAGE_KEY,
            key
        ].join(':'),
        limit: limit,
        search: search
    });
    var recentSearchesPlugin = _createRecentSearchesPlugin.createRecentSearchesPlugin({
        transformSource: transformSource,
        storage: storage,
        subscribe: subscribe
    });
    return _objectSpread(_objectSpread({}, recentSearchesPlugin), {}, {
        name: 'aa.localStorageRecentSearchesPlugin',
        __autocomplete_pluginOptions: options
    });
}
function getOptions(options) {
    return _objectSpread({
        limit: 5,
        search: _search.search,
        transformSource: function transformSource(_ref) {
            var source = _ref.source;
            return source;
        }
    }, options);
}

},{"./constants":"7gwd9","./createLocalStorage":"iPAw2","./createRecentSearchesPlugin":"2L1S7","./search":"CzUZD","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"7gwd9":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "LOCAL_STORAGE_KEY", ()=>LOCAL_STORAGE_KEY
);
parcelHelpers.export(exports, "LOCAL_STORAGE_KEY_TEST", ()=>LOCAL_STORAGE_KEY_TEST
);
var LOCAL_STORAGE_KEY = 'AUTOCOMPLETE_RECENT_SEARCHES';
var LOCAL_STORAGE_KEY_TEST = '__AUTOCOMPLETE_RECENT_SEARCHES_PLUGIN_TEST_KEY__';

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"iPAw2":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "createLocalStorage", ()=>createLocalStorage
);
var _getLocalStorage = require("./getLocalStorage");
function _toConsumableArray(arr) {
    return _arrayWithoutHoles(arr) || _iterableToArray(arr) || _unsupportedIterableToArray(arr) || _nonIterableSpread();
}
function _nonIterableSpread() {
    throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
}
function _unsupportedIterableToArray(o, minLen) {
    if (!o) return;
    if (typeof o === "string") return _arrayLikeToArray(o, minLen);
    var n = Object.prototype.toString.call(o).slice(8, -1);
    if (n === "Object" && o.constructor) n = o.constructor.name;
    if (n === "Map" || n === "Set") return Array.from(o);
    if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen);
}
function _iterableToArray(iter) {
    if (typeof Symbol !== "undefined" && iter[Symbol.iterator] != null || iter["@@iterator"] != null) return Array.from(iter);
}
function _arrayWithoutHoles(arr) {
    if (Array.isArray(arr)) return _arrayLikeToArray(arr);
}
function _arrayLikeToArray(arr, len) {
    if (len == null || len > arr.length) len = arr.length;
    for(var i = 0, arr2 = new Array(len); i < len; i++)arr2[i] = arr[i];
    return arr2;
}
function createLocalStorage(_ref) {
    var key = _ref.key, limit = _ref.limit, search = _ref.search;
    var storage = _getLocalStorage.getLocalStorage({
        key: key
    });
    return {
        onAdd: function onAdd(item) {
            storage.setItem([
                item
            ].concat(_toConsumableArray(storage.getItem())));
        },
        onRemove: function onRemove(id) {
            storage.setItem(storage.getItem().filter(function(x) {
                return x.id !== id;
            }));
        },
        getAll: function getAll() {
            var query = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : '';
            return search({
                query: query,
                items: storage.getItem(),
                limit: limit
            }).slice(0, limit);
        }
    };
}

},{"./getLocalStorage":"fSk8j","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"fSk8j":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "getLocalStorage", ()=>getLocalStorage
);
var _constants = require("./constants");
function isLocalStorageSupported() {
    try {
        localStorage.setItem(_constants.LOCAL_STORAGE_KEY_TEST, '');
        localStorage.removeItem(_constants.LOCAL_STORAGE_KEY_TEST);
        return true;
    } catch (error) {
        return false;
    }
}
function getLocalStorage(_ref) {
    var key = _ref.key;
    if (!isLocalStorageSupported()) return {
        setItem: function setItem() {},
        getItem: function getItem() {
            return [];
        }
    };
    return {
        setItem: function setItem(items) {
            return window.localStorage.setItem(key, JSON.stringify(items));
        },
        getItem: function getItem() {
            var items = window.localStorage.getItem(key);
            return items ? JSON.parse(items) : [];
        }
    };
}

},{"./constants":"7gwd9","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"2L1S7":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "createRecentSearchesPlugin", ()=>createRecentSearchesPlugin
);
var _autocompleteShared = require("@algolia/autocomplete-shared");
var _createStorageApi = require("./createStorageApi");
var _getTemplates = require("./getTemplates");
function _toConsumableArray(arr) {
    return _arrayWithoutHoles(arr) || _iterableToArray(arr) || _unsupportedIterableToArray(arr) || _nonIterableSpread();
}
function _nonIterableSpread() {
    throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
}
function _unsupportedIterableToArray(o, minLen) {
    if (!o) return;
    if (typeof o === "string") return _arrayLikeToArray(o, minLen);
    var n = Object.prototype.toString.call(o).slice(8, -1);
    if (n === "Object" && o.constructor) n = o.constructor.name;
    if (n === "Map" || n === "Set") return Array.from(o);
    if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen);
}
function _iterableToArray(iter) {
    if (typeof Symbol !== "undefined" && iter[Symbol.iterator] != null || iter["@@iterator"] != null) return Array.from(iter);
}
function _arrayWithoutHoles(arr) {
    if (Array.isArray(arr)) return _arrayLikeToArray(arr);
}
function _arrayLikeToArray(arr, len) {
    if (len == null || len > arr.length) len = arr.length;
    for(var i = 0, arr2 = new Array(len); i < len; i++)arr2[i] = arr[i];
    return arr2;
}
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        enumerableOnly && (symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        })), keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = null != arguments[i] ? arguments[i] : {};
        i % 2 ? ownKeys(Object(source), !0).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
function getDefaultSubcribe(store) {
    return function subscribe(_ref) {
        var onSelect = _ref.onSelect;
        onSelect(function(_ref2) {
            var item = _ref2.item, state = _ref2.state, source = _ref2.source;
            var inputValue = source.getItemInputValue({
                item: item,
                state: state
            });
            if (source.sourceId === 'querySuggestionsPlugin' && inputValue) {
                var recentItem = {
                    id: inputValue,
                    label: inputValue,
                    category: item.__autocomplete_qsCategory
                };
                store.addItem(recentItem);
            }
        });
    };
}
function createRecentSearchesPlugin(options) {
    var _getOptions = getOptions(options), storage = _getOptions.storage, transformSource = _getOptions.transformSource, subscribe = _getOptions.subscribe;
    var store = _createStorageApi.createStorageApi(storage);
    var lastItemsRef = _autocompleteShared.createRef([]);
    return {
        name: 'aa.recentSearchesPlugin',
        subscribe: subscribe !== null && subscribe !== void 0 ? subscribe : getDefaultSubcribe(store),
        onSubmit: function onSubmit(_ref3) {
            var state = _ref3.state;
            var query = state.query;
            if (query) {
                var recentItem = {
                    id: query,
                    label: query
                };
                store.addItem(recentItem);
            }
        },
        getSources: function getSources(_ref4) {
            var query = _ref4.query, setQuery = _ref4.setQuery, refresh = _ref4.refresh, state = _ref4.state;
            lastItemsRef.current = store.getAll(query);
            function onRemove(id) {
                store.removeItem(id);
                refresh();
            }
            function onTapAhead(item) {
                setQuery(item.label);
                refresh();
            }
            return Promise.resolve(lastItemsRef.current).then(function(items) {
                if (items.length === 0) return [];
                return [
                    transformSource({
                        source: {
                            sourceId: 'recentSearchesPlugin',
                            getItemInputValue: function getItemInputValue(_ref5) {
                                var item = _ref5.item;
                                return item.label;
                            },
                            getItems: function getItems() {
                                return items;
                            },
                            templates: _getTemplates.getTemplates({
                                onRemove: onRemove,
                                onTapAhead: onTapAhead
                            })
                        },
                        onRemove: onRemove,
                        onTapAhead: onTapAhead,
                        state: state
                    })
                ];
            });
        },
        data: _objectSpread(_objectSpread({}, store), {}, {
            // @ts-ignore SearchOptions `facetFilters` is ReadonlyArray
            getAlgoliaSearchParams: function getAlgoliaSearchParams() {
                var _params$facetFilters, _params$hitsPerPage;
                var params = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : {};
                // If the items returned by `store.getAll` are contained in a Promise,
                // we cannot provide the search params in time when this function is called
                // because we need to resolve the promise before getting the value.
                if (!Array.isArray(lastItemsRef.current)) {
                    _autocompleteShared.warn(false, 'The `getAlgoliaSearchParams` function is not supported with storages that return promises in `getAll`.');
                    return params;
                }
                return _objectSpread(_objectSpread({}, params), {}, {
                    facetFilters: [].concat(_toConsumableArray((_params$facetFilters = params.facetFilters) !== null && _params$facetFilters !== void 0 ? _params$facetFilters : []), _toConsumableArray(lastItemsRef.current.map(function(item) {
                        return [
                            "objectID:-".concat(item.label)
                        ];
                    }))),
                    hitsPerPage: Math.max(1, ((_params$hitsPerPage = params.hitsPerPage) !== null && _params$hitsPerPage !== void 0 ? _params$hitsPerPage : 10) - lastItemsRef.current.length)
                });
            }
        }),
        __autocomplete_pluginOptions: options
    };
}
function getOptions(options) {
    return _objectSpread({
        transformSource: function transformSource(_ref6) {
            var source = _ref6.source;
            return source;
        }
    }, options);
}

},{"@algolia/autocomplete-shared":"59T59","./createStorageApi":"gg2wy","./getTemplates":"8XaSk","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"gg2wy":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "createStorageApi", ()=>createStorageApi
);
function createStorageApi(storage) {
    return {
        addItem: function addItem(item) {
            storage.onRemove(item.id);
            storage.onAdd(item);
        },
        removeItem: function removeItem(id) {
            storage.onRemove(id);
        },
        getAll: function getAll(query) {
            return storage.getAll(query);
        }
    };
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"8XaSk":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/** @jsx createElement */ parcelHelpers.export(exports, "getTemplates", ()=>getTemplates
);
function getTemplates(_ref) {
    var onRemove = _ref.onRemove, onTapAhead = _ref.onTapAhead;
    return {
        item: function item(_ref2) {
            var item = _ref2.item, createElement = _ref2.createElement, components = _ref2.components;
            return createElement("div", {
                className: "aa-ItemWrapper"
            }, createElement("div", {
                className: "aa-ItemContent"
            }, createElement("div", {
                className: "aa-ItemIcon aa-ItemIcon--noBorder"
            }, createElement("svg", {
                viewBox: "0 0 24 24",
                fill: "currentColor"
            }, createElement("path", {
                d: "M12.516 6.984v5.25l4.5 2.672-0.75 1.266-5.25-3.188v-6h1.5zM12 20.016q3.281 0 5.648-2.367t2.367-5.648-2.367-5.648-5.648-2.367-5.648 2.367-2.367 5.648 2.367 5.648 5.648 2.367zM12 2.016q4.125 0 7.055 2.93t2.93 7.055-2.93 7.055-7.055 2.93-7.055-2.93-2.93-7.055 2.93-7.055 7.055-2.93z"
            }))), createElement("div", {
                className: "aa-ItemContentBody"
            }, createElement("div", {
                className: "aa-ItemContentTitle"
            }, createElement(components.ReverseHighlight, {
                hit: item,
                attribute: "label"
            }), item.category && createElement("span", {
                className: "aa-ItemContentSubtitle aa-ItemContentSubtitle--inline"
            }, createElement("span", {
                className: "aa-ItemContentSubtitleIcon"
            }), " in", ' ', createElement("span", {
                className: "aa-ItemContentSubtitleCategory"
            }, item.category))))), createElement("div", {
                className: "aa-ItemActions"
            }, createElement("button", {
                className: "aa-ItemActionButton",
                title: "Remove this search",
                onClick: function onClick(event) {
                    event.preventDefault();
                    event.stopPropagation();
                    onRemove(item.id);
                }
            }, createElement("svg", {
                viewBox: "0 0 24 24",
                fill: "currentColor"
            }, createElement("path", {
                d: "M18 7v13c0 0.276-0.111 0.525-0.293 0.707s-0.431 0.293-0.707 0.293h-10c-0.276 0-0.525-0.111-0.707-0.293s-0.293-0.431-0.293-0.707v-13zM17 5v-1c0-0.828-0.337-1.58-0.879-2.121s-1.293-0.879-2.121-0.879h-4c-0.828 0-1.58 0.337-2.121 0.879s-0.879 1.293-0.879 2.121v1h-4c-0.552 0-1 0.448-1 1s0.448 1 1 1h1v13c0 0.828 0.337 1.58 0.879 2.121s1.293 0.879 2.121 0.879h10c0.828 0 1.58-0.337 2.121-0.879s0.879-1.293 0.879-2.121v-13h1c0.552 0 1-0.448 1-1s-0.448-1-1-1zM9 5v-1c0-0.276 0.111-0.525 0.293-0.707s0.431-0.293 0.707-0.293h4c0.276 0 0.525 0.111 0.707 0.293s0.293 0.431 0.293 0.707v1zM9 11v6c0 0.552 0.448 1 1 1s1-0.448 1-1v-6c0-0.552-0.448-1-1-1s-1 0.448-1 1zM13 11v6c0 0.552 0.448 1 1 1s1-0.448 1-1v-6c0-0.552-0.448-1-1-1s-1 0.448-1 1z"
            }))), createElement("button", {
                className: "aa-ItemActionButton",
                title: "Fill query with \"".concat(item.label, "\""),
                onClick: function onClick(event) {
                    event.preventDefault();
                    event.stopPropagation();
                    onTapAhead(item);
                }
            }, createElement("svg", {
                viewBox: "0 0 24 24",
                fill: "currentColor"
            }, createElement("path", {
                d: "M8 17v-7.586l8.293 8.293c0.391 0.391 1.024 0.391 1.414 0s0.391-1.024 0-1.414l-8.293-8.293h7.586c0.552 0 1-0.448 1-1s-0.448-1-1-1h-10c-0.552 0-1 0.448-1 1v10c0 0.552 0.448 1 1 1s1-0.448 1-1z"
            })))));
        }
    };
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"CzUZD":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "search", ()=>search
);
var _addHighlightedAttribute = require("./addHighlightedAttribute");
function search(_ref) {
    var query = _ref.query, items = _ref.items, limit = _ref.limit;
    if (!query) return items.slice(0, limit).map(function(item) {
        return _addHighlightedAttribute.addHighlightedAttribute({
            item: item,
            query: query
        });
    });
    return items.filter(function(item) {
        return item.label.toLowerCase().includes(query.toLowerCase());
    }).slice(0, limit).map(function(item) {
        return _addHighlightedAttribute.addHighlightedAttribute({
            item: item,
            query: query
        });
    });
}

},{"./addHighlightedAttribute":"iMgYs","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"bk5Jd":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "clearRefinements", ()=>_clearRefinementsDefault.default
);
parcelHelpers.export(exports, "configure", ()=>_configureDefault.default
);
parcelHelpers.export(exports, "EXPERIMENTAL_configureRelatedItems", ()=>_configureRelatedItemsDefault.default
);
parcelHelpers.export(exports, "currentRefinements", ()=>_currentRefinementsDefault.default
);
parcelHelpers.export(exports, "geoSearch", ()=>_geoSearchDefault.default
);
parcelHelpers.export(exports, "hierarchicalMenu", ()=>_hierarchicalMenuDefault.default
);
parcelHelpers.export(exports, "hits", ()=>_hitsDefault.default
);
parcelHelpers.export(exports, "hitsPerPage", ()=>_hitsPerPageDefault.default
);
parcelHelpers.export(exports, "infiniteHits", ()=>_infiniteHitsDefault.default
);
parcelHelpers.export(exports, "menu", ()=>_menuDefault.default
);
parcelHelpers.export(exports, "refinementList", ()=>_refinementListDefault.default
);
parcelHelpers.export(exports, "numericMenu", ()=>_numericMenuDefault.default
);
parcelHelpers.export(exports, "pagination", ()=>_paginationDefault.default
);
parcelHelpers.export(exports, "rangeInput", ()=>_rangeInputDefault.default
);
parcelHelpers.export(exports, "searchBox", ()=>_searchBoxDefault.default
);
parcelHelpers.export(exports, "rangeSlider", ()=>_rangeSliderDefault.default
);
parcelHelpers.export(exports, "sortBy", ()=>_sortByDefault.default
);
parcelHelpers.export(exports, "ratingMenu", ()=>_ratingMenuDefault.default
);
parcelHelpers.export(exports, "stats", ()=>_statsDefault.default
);
parcelHelpers.export(exports, "toggleRefinement", ()=>_toggleRefinementDefault.default
);
parcelHelpers.export(exports, "analytics", ()=>_analyticsDefault.default
);
parcelHelpers.export(exports, "breadcrumb", ()=>_breadcrumbDefault.default
);
parcelHelpers.export(exports, "menuSelect", ()=>_menuSelectDefault.default
);
parcelHelpers.export(exports, "poweredBy", ()=>_poweredByDefault.default
);
parcelHelpers.export(exports, "panel", ()=>_panelDefault.default
);
parcelHelpers.export(exports, "voiceSearch", ()=>_voiceSearchDefault.default
);
parcelHelpers.export(exports, "queryRuleCustomData", ()=>_queryRuleCustomDataDefault.default
);
parcelHelpers.export(exports, "queryRuleContext", ()=>_queryRuleContextDefault.default
);
parcelHelpers.export(exports, "index", ()=>_indexDefault.default
);
parcelHelpers.export(exports, "places", ()=>_placesDefault.default
);
parcelHelpers.export(exports, "EXPERIMENTAL_answers", ()=>_answersDefault.default
);
parcelHelpers.export(exports, "relevantSort", ()=>_relevantSortDefault.default
);
parcelHelpers.export(exports, "EXPERIMENTAL_dynamicWidgets", ()=>_dynamicWidgetsDefault.default
);
var _clearRefinements = require("./clear-refinements/clear-refinements");
var _clearRefinementsDefault = parcelHelpers.interopDefault(_clearRefinements);
var _configure = require("./configure/configure");
var _configureDefault = parcelHelpers.interopDefault(_configure);
var _configureRelatedItems = require("./configure-related-items/configure-related-items");
var _configureRelatedItemsDefault = parcelHelpers.interopDefault(_configureRelatedItems);
var _currentRefinements = require("./current-refinements/current-refinements");
var _currentRefinementsDefault = parcelHelpers.interopDefault(_currentRefinements);
var _geoSearch = require("./geo-search/geo-search");
var _geoSearchDefault = parcelHelpers.interopDefault(_geoSearch);
var _hierarchicalMenu = require("./hierarchical-menu/hierarchical-menu");
var _hierarchicalMenuDefault = parcelHelpers.interopDefault(_hierarchicalMenu);
var _hits = require("./hits/hits");
var _hitsDefault = parcelHelpers.interopDefault(_hits);
var _hitsPerPage = require("./hits-per-page/hits-per-page");
var _hitsPerPageDefault = parcelHelpers.interopDefault(_hitsPerPage);
var _infiniteHits = require("./infinite-hits/infinite-hits");
var _infiniteHitsDefault = parcelHelpers.interopDefault(_infiniteHits);
var _menu = require("./menu/menu");
var _menuDefault = parcelHelpers.interopDefault(_menu);
var _refinementList = require("./refinement-list/refinement-list");
var _refinementListDefault = parcelHelpers.interopDefault(_refinementList);
var _numericMenu = require("./numeric-menu/numeric-menu");
var _numericMenuDefault = parcelHelpers.interopDefault(_numericMenu);
var _pagination = require("./pagination/pagination");
var _paginationDefault = parcelHelpers.interopDefault(_pagination);
var _rangeInput = require("./range-input/range-input");
var _rangeInputDefault = parcelHelpers.interopDefault(_rangeInput);
var _searchBox = require("./search-box/search-box");
var _searchBoxDefault = parcelHelpers.interopDefault(_searchBox);
var _rangeSlider = require("./range-slider/range-slider");
var _rangeSliderDefault = parcelHelpers.interopDefault(_rangeSlider);
var _sortBy = require("./sort-by/sort-by");
var _sortByDefault = parcelHelpers.interopDefault(_sortBy);
var _ratingMenu = require("./rating-menu/rating-menu");
var _ratingMenuDefault = parcelHelpers.interopDefault(_ratingMenu);
var _stats = require("./stats/stats");
var _statsDefault = parcelHelpers.interopDefault(_stats);
var _toggleRefinement = require("./toggle-refinement/toggle-refinement");
var _toggleRefinementDefault = parcelHelpers.interopDefault(_toggleRefinement);
var _analytics = require("./analytics/analytics");
var _analyticsDefault = parcelHelpers.interopDefault(_analytics);
var _breadcrumb = require("./breadcrumb/breadcrumb");
var _breadcrumbDefault = parcelHelpers.interopDefault(_breadcrumb);
var _menuSelect = require("./menu-select/menu-select");
var _menuSelectDefault = parcelHelpers.interopDefault(_menuSelect);
var _poweredBy = require("./powered-by/powered-by");
var _poweredByDefault = parcelHelpers.interopDefault(_poweredBy);
var _panel = require("./panel/panel");
var _panelDefault = parcelHelpers.interopDefault(_panel);
var _voiceSearch = require("./voice-search/voice-search");
var _voiceSearchDefault = parcelHelpers.interopDefault(_voiceSearch);
var _queryRuleCustomData = require("./query-rule-custom-data/query-rule-custom-data");
var _queryRuleCustomDataDefault = parcelHelpers.interopDefault(_queryRuleCustomData);
var _queryRuleContext = require("./query-rule-context/query-rule-context");
var _queryRuleContextDefault = parcelHelpers.interopDefault(_queryRuleContext);
var _index = require("./index/index");
var _indexDefault = parcelHelpers.interopDefault(_index);
var _places = require("./places/places");
var _placesDefault = parcelHelpers.interopDefault(_places);
var _answers = require("./answers/answers");
var _answersDefault = parcelHelpers.interopDefault(_answers);
var _relevantSort = require("./relevant-sort/relevant-sort");
var _relevantSortDefault = parcelHelpers.interopDefault(_relevantSort);
var _dynamicWidgets = require("./dynamic-widgets/dynamic-widgets");
var _dynamicWidgetsDefault = parcelHelpers.interopDefault(_dynamicWidgets);

},{"./clear-refinements/clear-refinements":false,"./configure/configure":"gLYAR","./configure-related-items/configure-related-items":false,"./current-refinements/current-refinements":false,"./geo-search/geo-search":false,"./hierarchical-menu/hierarchical-menu":false,"./hits/hits":"bPDYG","./hits-per-page/hits-per-page":false,"./infinite-hits/infinite-hits":false,"./menu/menu":false,"./refinement-list/refinement-list":"5rn2R","./numeric-menu/numeric-menu":false,"./pagination/pagination":"aGC8J","./range-input/range-input":false,"./search-box/search-box":false,"./range-slider/range-slider":"4PvhT","./sort-by/sort-by":"2sKTT","./rating-menu/rating-menu":false,"./stats/stats":false,"./toggle-refinement/toggle-refinement":false,"./analytics/analytics":false,"./breadcrumb/breadcrumb":false,"./menu-select/menu-select":false,"./powered-by/powered-by":false,"./panel/panel":false,"./voice-search/voice-search":false,"./query-rule-custom-data/query-rule-custom-data":false,"./query-rule-context/query-rule-context":false,"./index/index":"kdZTz","./places/places":false,"./answers/answers":false,"./relevant-sort/relevant-sort":false,"./dynamic-widgets/dynamic-widgets":false,"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"gLYAR":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _connectConfigure = require("../../connectors/configure/connectConfigure");
var _connectConfigureDefault = parcelHelpers.interopDefault(_connectConfigure);
var _utils = require("../../lib/utils");
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        if (enumerableOnly) symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        });
        keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = arguments[i] != null ? arguments[i] : {};
        if (i % 2) ownKeys(Object(source), true).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        });
        else if (Object.getOwnPropertyDescriptors) Object.defineProperties(target, Object.getOwnPropertyDescriptors(source));
        else ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
/**
 * A list of [search parameters](https://www.algolia.com/doc/api-reference/search-api-parameters/)
 * to enable when the widget mounts.
 */ var configure = function configure(widgetParams) {
    // This is a renderless widget that falls back to the connector's
    // noop render and unmount functions.
    var makeWidget = _connectConfigureDefault.default(_utils.noop);
    return _objectSpread(_objectSpread({}, makeWidget({
        searchParameters: widgetParams
    })), {}, {
        $$widgetType: 'ais.configure'
    });
};
exports.default = configure;

},{"../../connectors/configure/connectConfigure":"lvgHS","../../lib/utils":"etVYs","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"bPDYG":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/** @jsx h */ var _preact = require("preact");
var _classnames = require("classnames");
var _classnamesDefault = parcelHelpers.interopDefault(_classnames);
var _connectHits = require("../../connectors/hits/connectHits");
var _connectHitsDefault = parcelHelpers.interopDefault(_connectHits);
var _hits = require("../../components/Hits/Hits");
var _hitsDefault = parcelHelpers.interopDefault(_hits);
var _defaultTemplates = require("./defaultTemplates");
var _defaultTemplatesDefault = parcelHelpers.interopDefault(_defaultTemplates);
var _utils = require("../../lib/utils");
var _suit = require("../../lib/suit");
var _insights = require("../../lib/insights");
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        if (enumerableOnly) symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        });
        keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = arguments[i] != null ? arguments[i] : {};
        if (i % 2) ownKeys(Object(source), true).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        });
        else if (Object.getOwnPropertyDescriptors) Object.defineProperties(target, Object.getOwnPropertyDescriptors(source));
        else ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
var withUsage = _utils.createDocumentationMessageGenerator({
    name: 'hits'
});
var suit = _suit.component('Hits');
var HitsWithInsightsListener = _insights.withInsightsListener(_hitsDefault.default);
var renderer = function renderer(_ref) {
    var renderState = _ref.renderState, cssClasses = _ref.cssClasses, containerNode = _ref.containerNode, templates = _ref.templates;
    return function(_ref2, isFirstRendering) {
        var receivedHits = _ref2.hits, results = _ref2.results, instantSearchInstance = _ref2.instantSearchInstance, insights = _ref2.insights, bindEvent = _ref2.bindEvent;
        if (isFirstRendering) {
            renderState.templateProps = _utils.prepareTemplateProps({
                defaultTemplates: _defaultTemplatesDefault.default,
                templatesConfig: instantSearchInstance.templatesConfig,
                templates: templates
            });
            return;
        }
        _preact.render(_preact.h(HitsWithInsightsListener, {
            cssClasses: cssClasses,
            hits: receivedHits,
            results: results,
            templateProps: renderState.templateProps,
            insights: insights,
            sendEvent: function sendEvent(event) {
                instantSearchInstance.sendEventToInsights(event);
            },
            bindEvent: bindEvent
        }), containerNode);
    };
};
var hits = function hits(widgetParams) {
    var _ref3 = widgetParams || {}, container = _ref3.container, escapeHTML = _ref3.escapeHTML, transformItems = _ref3.transformItems, _ref3$templates = _ref3.templates, templates = _ref3$templates === void 0 ? _defaultTemplatesDefault.default : _ref3$templates, _ref3$cssClasses = _ref3.cssClasses, userCssClasses = _ref3$cssClasses === void 0 ? {} : _ref3$cssClasses;
    if (!container) throw new Error(withUsage('The `container` option is required.'));
    var containerNode = _utils.getContainerNode(container);
    var cssClasses = {
        root: _classnamesDefault.default(suit(), userCssClasses.root),
        emptyRoot: _classnamesDefault.default(suit({
            modifierName: 'empty'
        }), userCssClasses.emptyRoot),
        list: _classnamesDefault.default(suit({
            descendantName: 'list'
        }), userCssClasses.list),
        item: _classnamesDefault.default(suit({
            descendantName: 'item'
        }), userCssClasses.item)
    };
    var specializedRenderer = renderer({
        containerNode: containerNode,
        cssClasses: cssClasses,
        renderState: {},
        templates: templates
    });
    var makeWidget = _insights.withInsights(_connectHitsDefault.default)(specializedRenderer, function() {
        return _preact.render(null, containerNode);
    });
    return _objectSpread(_objectSpread({}, makeWidget({
        escapeHTML: escapeHTML,
        transformItems: transformItems
    })), {}, {
        $$widgetType: 'ais.hits'
    });
};
exports.default = hits;

},{"preact":"26zcy","classnames":"jocGM","../../connectors/hits/connectHits":"b5DNx","../../components/Hits/Hits":"as3BB","./defaultTemplates":"fxjDh","../../lib/utils":"etVYs","../../lib/suit":"du81D","../../lib/insights":"hnOzt","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"jocGM":[function(require,module,exports) {
/*!
  Copyright (c) 2018 Jed Watson.
  Licensed under the MIT License (MIT), see
  http://jedwatson.github.io/classnames
*/ /* global define */ (function() {
    var hasOwn = {}.hasOwnProperty;
    function classNames() {
        var classes = [];
        for(var i = 0; i < arguments.length; i++){
            var arg = arguments[i];
            if (!arg) continue;
            var argType = typeof arg;
            if (argType === 'string' || argType === 'number') classes.push(arg);
            else if (Array.isArray(arg)) {
                if (arg.length) {
                    var inner = classNames.apply(null, arg);
                    if (inner) classes.push(inner);
                }
            } else if (argType === 'object') {
                if (arg.toString === Object.prototype.toString) {
                    for(var key in arg)if (hasOwn.call(arg, key) && arg[key]) classes.push(key);
                } else classes.push(arg.toString());
            }
        }
        return classes.join(' ');
    }
    if (module.exports) {
        classNames.default = classNames;
        module.exports = classNames;
    } else if (typeof define === 'function' && typeof define.amd === 'object' && define.amd) // register as 'classnames', consistent with npm package name
    define('classnames', [], function() {
        return classNames;
    });
    else window.classNames = classNames;
})();

},{}],"as3BB":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/** @jsx h */ var _preact = require("preact");
var _classnames = require("classnames");
var _classnamesDefault = parcelHelpers.interopDefault(_classnames);
var _template = require("../Template/Template");
var _templateDefault = parcelHelpers.interopDefault(_template);
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        if (enumerableOnly) symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        });
        keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = arguments[i] != null ? arguments[i] : {};
        if (i % 2) ownKeys(Object(source), true).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        });
        else if (Object.getOwnPropertyDescriptors) Object.defineProperties(target, Object.getOwnPropertyDescriptors(source));
        else ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
function _extends() {
    _extends = Object.assign || function(target) {
        for(var i = 1; i < arguments.length; i++){
            var source = arguments[i];
            for(var key in source)if (Object.prototype.hasOwnProperty.call(source, key)) target[key] = source[key];
        }
        return target;
    };
    return _extends.apply(this, arguments);
}
var Hits = function Hits(_ref) {
    var results = _ref.results, hits = _ref.hits, bindEvent = _ref.bindEvent, cssClasses = _ref.cssClasses, templateProps = _ref.templateProps;
    if (results.hits.length === 0) return _preact.h(_templateDefault.default, _extends({}, templateProps, {
        templateKey: "empty",
        rootProps: {
            className: _classnamesDefault.default(cssClasses.root, cssClasses.emptyRoot)
        },
        data: results
    }));
    return _preact.h("div", {
        className: cssClasses.root
    }, _preact.h("ol", {
        className: cssClasses.list
    }, hits.map(function(hit, position) {
        return _preact.h(_templateDefault.default, _extends({}, templateProps, {
            templateKey: "item",
            rootTagName: "li",
            rootProps: {
                className: cssClasses.item
            },
            key: hit.objectID,
            data: _objectSpread(_objectSpread({}, hit), {}, {
                __hitIndex: position
            }),
            bindEvent: bindEvent
        }));
    })));
};
Hits.defaultProps = {
    results: {
        hits: []
    },
    hits: []
};
exports.default = Hits;

},{"preact":"26zcy","classnames":"jocGM","../Template/Template":"aVPg5","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"aVPg5":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/** @jsx h */ var _preact = require("preact");
var _utils = require("../../lib/utils");
function _typeof(obj1) {
    if (typeof Symbol === "function" && typeof Symbol.iterator === "symbol") _typeof = function _typeof(obj) {
        return typeof obj;
    };
    else _typeof = function _typeof(obj) {
        return obj && typeof Symbol === "function" && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj;
    };
    return _typeof(obj1);
}
function _extends() {
    _extends = Object.assign || function(target) {
        for(var i = 1; i < arguments.length; i++){
            var source = arguments[i];
            for(var key in source)if (Object.prototype.hasOwnProperty.call(source, key)) target[key] = source[key];
        }
        return target;
    };
    return _extends.apply(this, arguments);
}
function _classCallCheck(instance, Constructor) {
    if (!(instance instanceof Constructor)) throw new TypeError("Cannot call a class as a function");
}
function _defineProperties(target, props) {
    for(var i = 0; i < props.length; i++){
        var descriptor = props[i];
        descriptor.enumerable = descriptor.enumerable || false;
        descriptor.configurable = true;
        if ("value" in descriptor) descriptor.writable = true;
        Object.defineProperty(target, descriptor.key, descriptor);
    }
}
function _createClass(Constructor, protoProps, staticProps) {
    if (protoProps) _defineProperties(Constructor.prototype, protoProps);
    if (staticProps) _defineProperties(Constructor, staticProps);
    return Constructor;
}
function _inherits(subClass, superClass) {
    if (typeof superClass !== "function" && superClass !== null) throw new TypeError("Super expression must either be null or a function");
    subClass.prototype = Object.create(superClass && superClass.prototype, {
        constructor: {
            value: subClass,
            writable: true,
            configurable: true
        }
    });
    if (superClass) _setPrototypeOf(subClass, superClass);
}
function _setPrototypeOf(o1, p1) {
    _setPrototypeOf = Object.setPrototypeOf || function _setPrototypeOf(o, p) {
        o.__proto__ = p;
        return o;
    };
    return _setPrototypeOf(o1, p1);
}
function _createSuper(Derived) {
    var hasNativeReflectConstruct = _isNativeReflectConstruct();
    return function _createSuperInternal() {
        var Super = _getPrototypeOf(Derived), result;
        if (hasNativeReflectConstruct) {
            var NewTarget = _getPrototypeOf(this).constructor;
            result = Reflect.construct(Super, arguments, NewTarget);
        } else result = Super.apply(this, arguments);
        return _possibleConstructorReturn(this, result);
    };
}
function _possibleConstructorReturn(self, call) {
    if (call && (_typeof(call) === "object" || typeof call === "function")) return call;
    return _assertThisInitialized(self);
}
function _assertThisInitialized(self) {
    if (self === void 0) throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
    return self;
}
function _isNativeReflectConstruct() {
    if (typeof Reflect === "undefined" || !Reflect.construct) return false;
    if (Reflect.construct.sham) return false;
    if (typeof Proxy === "function") return true;
    try {
        Date.prototype.toString.call(Reflect.construct(Date, [], function() {}));
        return true;
    } catch (e) {
        return false;
    }
}
function _getPrototypeOf(o2) {
    _getPrototypeOf = Object.setPrototypeOf ? Object.getPrototypeOf : function _getPrototypeOf(o) {
        return o.__proto__ || Object.getPrototypeOf(o);
    };
    return _getPrototypeOf(o2);
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
var defaultProps = {
    data: {},
    rootTagName: 'div',
    useCustomCompileOptions: {},
    templates: {},
    templatesConfig: {}
};
// @TODO: Template should be a generic and receive TData to pass to Templates (to avoid TTemplateData to be set as `any`)
var Template = /*#__PURE__*/ function(_Component) {
    _inherits(Template1, _Component);
    var _super = _createSuper(Template1);
    function Template1() {
        _classCallCheck(this, Template1);
        return _super.apply(this, arguments);
    }
    _createClass(Template1, [
        {
            key: "shouldComponentUpdate",
            value: function shouldComponentUpdate(nextProps) {
                return !_utils.isEqual(this.props.data, nextProps.data) || this.props.templateKey !== nextProps.templateKey || !_utils.isEqual(this.props.rootProps, nextProps.rootProps);
            }
        },
        {
            key: "render",
            value: function render() {
                var RootTagName = this.props.rootTagName;
                var useCustomCompileOptions = this.props.useCustomCompileOptions[this.props.templateKey];
                var compileOptions = useCustomCompileOptions ? this.props.templatesConfig.compileOptions : {};
                var content = _utils.renderTemplate({
                    templates: this.props.templates,
                    templateKey: this.props.templateKey,
                    compileOptions: compileOptions,
                    helpers: this.props.templatesConfig.helpers,
                    data: this.props.data,
                    bindEvent: this.props.bindEvent
                });
                if (content === null) // Adds a noscript to the DOM but virtual DOM is null
                // See http://facebook.github.io/react/docs/component-specs.html#render
                return null;
                return _preact.h(RootTagName, _extends({}, this.props.rootProps, {
                    dangerouslySetInnerHTML: {
                        __html: content
                    }
                }));
            }
        }
    ]);
    return Template1;
}(_preact.Component);
_defineProperty(Template, "defaultProps", defaultProps);
exports.default = Template;

},{"preact":"26zcy","../../lib/utils":"etVYs","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"fxjDh":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
exports.default = {
    empty: 'No results',
    item: function item(data) {
        return JSON.stringify(data, null, 2);
    }
};

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"hnOzt":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "withInsights", ()=>_clientDefault.default
);
parcelHelpers.export(exports, "inferInsightsPayload", ()=>_client.inferPayload
);
parcelHelpers.export(exports, "withInsightsListener", ()=>_listenerDefault.default
);
var _client = require("./client");
var _clientDefault = parcelHelpers.interopDefault(_client);
var _listener = require("./listener");
var _listenerDefault = parcelHelpers.interopDefault(_listener);

},{"./client":"1CWch","./listener":"hhB68","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"1CWch":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "inferPayload", ()=>inferPayload
);
var _utils = require("../utils");
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        if (enumerableOnly) symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        });
        keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = arguments[i] != null ? arguments[i] : {};
        if (i % 2) ownKeys(Object(source), true).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        });
        else if (Object.getOwnPropertyDescriptors) Object.defineProperties(target, Object.getOwnPropertyDescriptors(source));
        else ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
var getSelectedHits = function getSelectedHits(hits, selectedObjectIDs) {
    return selectedObjectIDs.map(function(objectID) {
        var hit = _utils.find(hits, function(h) {
            return h.objectID === objectID;
        });
        if (typeof hit === 'undefined') throw new Error("Could not find objectID \"".concat(objectID, "\" passed to `clickedObjectIDsAfterSearch` in the returned hits. This is necessary to infer the absolute position and the query ID."));
        return hit;
    });
};
var getQueryID = function getQueryID(selectedHits) {
    var queryIDs = _utils.uniq(selectedHits.map(function(hit) {
        return hit.__queryID;
    }));
    if (queryIDs.length > 1) throw new Error('Insights currently allows a single `queryID`. The `objectIDs` provided map to multiple `queryID`s.');
    var queryID = queryIDs[0];
    if (typeof queryID !== 'string') throw new Error("Could not infer `queryID`. Ensure InstantSearch `clickAnalytics: true` was added with the Configure widget.\n\nSee: https://alg.li/lNiZZ7");
    return queryID;
};
var getPositions = function getPositions(selectedHits) {
    return selectedHits.map(function(hit) {
        return hit.__position;
    });
};
var inferPayload = function inferPayload(_ref) {
    var method = _ref.method, results = _ref.results, hits = _ref.hits, objectIDs = _ref.objectIDs;
    var index = results.index;
    var selectedHits = getSelectedHits(hits, objectIDs);
    var queryID = getQueryID(selectedHits);
    switch(method){
        case 'clickedObjectIDsAfterSearch':
            var positions = getPositions(selectedHits);
            return {
                index: index,
                queryID: queryID,
                objectIDs: objectIDs,
                positions: positions
            };
        case 'convertedObjectIDsAfterSearch':
            return {
                index: index,
                queryID: queryID,
                objectIDs: objectIDs
            };
        default:
            throw new Error("Unsupported method passed to insights: \"".concat(method, "\"."));
    }
};
var wrapInsightsClient = function wrapInsightsClient(aa, results, hits) {
    return function(method, payload) {
        _utils.warning(false, "`insights` function has been deprecated. It is still supported in 4.x releases, but not further. It is replaced by the `insights` middleware.\n\nFor more information, visit https://www.algolia.com/doc/guides/getting-insights-and-analytics/search-analytics/click-through-and-conversions/how-to/send-click-and-conversion-events-with-instantsearch/js/");
        if (!aa) {
            var withInstantSearchUsage = _utils.createDocumentationMessageGenerator({
                name: 'instantsearch'
            });
            throw new Error(withInstantSearchUsage('The `insightsClient` option has not been provided to `instantsearch`.'));
        }
        if (!Array.isArray(payload.objectIDs)) throw new TypeError('Expected `objectIDs` to be an array.');
        var inferredPayload = inferPayload({
            method: method,
            results: results,
            hits: hits,
            objectIDs: payload.objectIDs
        });
        aa(method, _objectSpread(_objectSpread({}, inferredPayload), payload));
    };
};
function withInsights(connector) {
    return function(renderFn, unmountFn) {
        return connector(function(renderOptions, isFirstRender) {
            var results = renderOptions.results, hits = renderOptions.hits, instantSearchInstance = renderOptions.instantSearchInstance;
            if (results && hits && instantSearchInstance) {
                var insights = wrapInsightsClient(instantSearchInstance.insightsClient, results, hits);
                return renderFn(_objectSpread(_objectSpread({}, renderOptions), {}, {
                    insights: insights
                }), isFirstRender);
            }
            return renderFn(renderOptions, isFirstRender);
        }, unmountFn);
    };
}
exports.default = withInsights;

},{"../utils":"etVYs","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"hhB68":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/** @jsx h */ var _preact = require("preact");
var _utils = require("../utils");
var _insights = require("../../helpers/insights");
var findInsightsTarget = function findInsightsTarget(startElement, endElement, validator) {
    var element = startElement;
    while(element && !validator(element)){
        if (element === endElement) return null;
        element = element.parentElement;
    }
    return element;
};
var parseInsightsEvent = function parseInsightsEvent(element) {
    var serializedPayload = element.getAttribute('data-insights-event');
    if (typeof serializedPayload !== 'string') throw new Error('The insights middleware expects `data-insights-event` to be a base64-encoded JSON string.');
    try {
        return _utils.deserializePayload(serializedPayload);
    } catch (error) {
        throw new Error('The insights middleware was unable to parse `data-insights-event`.');
    }
};
var insightsListener = function insightsListener(BaseComponent) {
    function WithInsightsListener(props) {
        var handleClick = function handleClick(event) {
            if (props.sendEvent) {
                // new way with insights middleware
                var targetWithEvent = findInsightsTarget(event.target, event.currentTarget, function(element) {
                    return element.hasAttribute('data-insights-event');
                });
                if (targetWithEvent) {
                    var payload = parseInsightsEvent(targetWithEvent);
                    props.sendEvent(payload);
                }
            } // old way, e.g. instantsearch.insights("clickedObjectIDsAfterSearch", { .. })
            var insightsTarget = findInsightsTarget(event.target, event.currentTarget, function(element) {
                return _insights.hasDataAttributes(element);
            });
            if (insightsTarget) {
                var _readDataAttributes = _insights.readDataAttributes(insightsTarget), method = _readDataAttributes.method, _payload = _readDataAttributes.payload;
                props.insights(method, _payload);
            }
        };
        return _preact.h("div", {
            onClick: handleClick
        }, _preact.h(BaseComponent, props));
    }
    return WithInsightsListener;
};
exports.default = insightsListener;

},{"preact":"26zcy","../utils":"etVYs","../../helpers/insights":"2EZr9","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"5rn2R":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "defaultTemplates", ()=>defaultTemplates
);
/** @jsx h */ var _preact = require("preact");
var _classnames = require("classnames");
var _classnamesDefault = parcelHelpers.interopDefault(_classnames);
var _refinementList = require("../../components/RefinementList/RefinementList");
var _refinementListDefault = parcelHelpers.interopDefault(_refinementList);
var _connectRefinementList = require("../../connectors/refinement-list/connectRefinementList");
var _connectRefinementListDefault = parcelHelpers.interopDefault(_connectRefinementList);
var _utils = require("../../lib/utils");
var _suit = require("../../lib/suit");
var _defaultTemplates = require("../search-box/defaultTemplates");
var _defaultTemplatesDefault = parcelHelpers.interopDefault(_defaultTemplates);
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        if (enumerableOnly) symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        });
        keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = arguments[i] != null ? arguments[i] : {};
        if (i % 2) ownKeys(Object(source), true).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        });
        else if (Object.getOwnPropertyDescriptors) Object.defineProperties(target, Object.getOwnPropertyDescriptors(source));
        else ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
var withUsage = _utils.createDocumentationMessageGenerator({
    name: 'refinement-list'
});
var suit = _suit.component('RefinementList');
var searchBoxSuit = _suit.component('SearchBox');
var defaultTemplates = {
    item: "<label class=\"{{cssClasses.label}}\">\n  <input type=\"checkbox\"\n         class=\"{{cssClasses.checkbox}}\"\n         value=\"{{value}}\"\n         {{#isRefined}}checked{{/isRefined}} />\n  <span class=\"{{cssClasses.labelText}}\">{{#isFromSearch}}{{{highlighted}}}{{/isFromSearch}}{{^isFromSearch}}{{highlighted}}{{/isFromSearch}}</span>\n  <span class=\"{{cssClasses.count}}\">{{#helpers.formatNumber}}{{count}}{{/helpers.formatNumber}}</span>\n</label>",
    showMoreText: "\n    {{#isShowingMore}}\n      Show less\n    {{/isShowingMore}}\n    {{^isShowingMore}}\n      Show more\n    {{/isShowingMore}}\n    ",
    searchableNoResults: 'No results'
};
var renderer = function renderer(_ref) {
    var containerNode = _ref.containerNode, cssClasses = _ref.cssClasses, templates = _ref.templates, searchBoxTemplates = _ref.searchBoxTemplates, renderState = _ref.renderState, showMore = _ref.showMore, searchable = _ref.searchable, searchablePlaceholder = _ref.searchablePlaceholder, searchableIsAlwaysActive = _ref.searchableIsAlwaysActive;
    return function(_ref2, isFirstRendering) {
        var refine = _ref2.refine, items = _ref2.items, createURL = _ref2.createURL, searchForItems = _ref2.searchForItems, isFromSearch = _ref2.isFromSearch, instantSearchInstance = _ref2.instantSearchInstance, toggleShowMore = _ref2.toggleShowMore, isShowingMore = _ref2.isShowingMore, hasExhaustiveItems = _ref2.hasExhaustiveItems, canToggleShowMore = _ref2.canToggleShowMore;
        if (isFirstRendering) {
            renderState.templateProps = _utils.prepareTemplateProps({
                defaultTemplates: defaultTemplates,
                templatesConfig: instantSearchInstance.templatesConfig,
                templates: templates
            });
            renderState.searchBoxTemplateProps = _utils.prepareTemplateProps({
                defaultTemplates: _defaultTemplatesDefault.default,
                templatesConfig: instantSearchInstance.templatesConfig,
                templates: searchBoxTemplates
            });
            return;
        }
        _preact.render(_preact.h(_refinementListDefault.default, {
            createURL: createURL,
            cssClasses: cssClasses,
            facetValues: items,
            templateProps: renderState.templateProps,
            searchBoxTemplateProps: renderState.searchBoxTemplateProps,
            toggleRefinement: refine,
            searchFacetValues: searchable ? searchForItems : undefined,
            searchPlaceholder: searchablePlaceholder,
            searchIsAlwaysActive: searchableIsAlwaysActive,
            isFromSearch: isFromSearch,
            showMore: showMore && !isFromSearch && items.length > 0,
            toggleShowMore: toggleShowMore,
            isShowingMore: isShowingMore,
            hasExhaustiveItems: hasExhaustiveItems,
            canToggleShowMore: canToggleShowMore
        }), containerNode);
    };
};
/**
 * The refinement list widget is one of the most common widget that you can find
 * in a search UI. With this widget, the user can filter the dataset based on facets.
 *
 * The refinement list displays only the most relevant facets for the current search
 * context. The sort option only affects the facet that are returned by the engine,
 * not which facets are returned.
 *
 * This widget also implements search for facet values, which is a mini search inside the
 * values of the facets. This makes easy to deal with uncommon facet values.
 *
 * @requirements
 *
 * The attribute passed to `attribute` must be declared as an
 * [attribute for faceting](https://www.algolia.com/doc/guides/searching/faceting/#declaring-attributes-for-faceting)
 * in your Algolia settings.
 *
 * If you also want to use search for facet values on this attribute, you need to make it searchable using the [dashboard](https://www.algolia.com/explorer/display/) or using the [API](https://www.algolia.com/doc/guides/searching/faceting/#search-for-facet-values).
 */ var refinementList = function refinementList(widgetParams) {
    var _ref3 = widgetParams || {}, container = _ref3.container, attribute = _ref3.attribute, operator = _ref3.operator, sortBy = _ref3.sortBy, limit = _ref3.limit, showMore = _ref3.showMore, showMoreLimit = _ref3.showMoreLimit, _ref3$searchable = _ref3.searchable, searchable = _ref3$searchable === void 0 ? false : _ref3$searchable, _ref3$searchablePlace = _ref3.searchablePlaceholder, searchablePlaceholder = _ref3$searchablePlace === void 0 ? 'Search...' : _ref3$searchablePlace, _ref3$searchableEscap = _ref3.searchableEscapeFacetValues, searchableEscapeFacetValues = _ref3$searchableEscap === void 0 ? true : _ref3$searchableEscap, _ref3$searchableIsAlw = _ref3.searchableIsAlwaysActive, searchableIsAlwaysActive = _ref3$searchableIsAlw === void 0 ? true : _ref3$searchableIsAlw, _ref3$cssClasses = _ref3.cssClasses, userCssClasses = _ref3$cssClasses === void 0 ? {} : _ref3$cssClasses, _ref3$templates = _ref3.templates, userTemplates = _ref3$templates === void 0 ? defaultTemplates : _ref3$templates, transformItems = _ref3.transformItems;
    if (!container) throw new Error(withUsage('The `container` option is required.'));
    var escapeFacetValues = searchable ? Boolean(searchableEscapeFacetValues) : false;
    var containerNode = _utils.getContainerNode(container);
    var cssClasses = {
        root: _classnamesDefault.default(suit(), userCssClasses.root),
        noRefinementRoot: _classnamesDefault.default(suit({
            modifierName: 'noRefinement'
        }), userCssClasses.noRefinementRoot),
        list: _classnamesDefault.default(suit({
            descendantName: 'list'
        }), userCssClasses.list),
        item: _classnamesDefault.default(suit({
            descendantName: 'item'
        }), userCssClasses.item),
        selectedItem: _classnamesDefault.default(suit({
            descendantName: 'item',
            modifierName: 'selected'
        }), userCssClasses.selectedItem),
        searchBox: _classnamesDefault.default(suit({
            descendantName: 'searchBox'
        }), userCssClasses.searchBox),
        label: _classnamesDefault.default(suit({
            descendantName: 'label'
        }), userCssClasses.label),
        checkbox: _classnamesDefault.default(suit({
            descendantName: 'checkbox'
        }), userCssClasses.checkbox),
        labelText: _classnamesDefault.default(suit({
            descendantName: 'labelText'
        }), userCssClasses.labelText),
        count: _classnamesDefault.default(suit({
            descendantName: 'count'
        }), userCssClasses.count),
        noResults: _classnamesDefault.default(suit({
            descendantName: 'noResults'
        }), userCssClasses.noResults),
        showMore: _classnamesDefault.default(suit({
            descendantName: 'showMore'
        }), userCssClasses.showMore),
        disabledShowMore: _classnamesDefault.default(suit({
            descendantName: 'showMore',
            modifierName: 'disabled'
        }), userCssClasses.disabledShowMore),
        searchable: {
            root: _classnamesDefault.default(searchBoxSuit(), userCssClasses.searchableRoot),
            form: _classnamesDefault.default(searchBoxSuit({
                descendantName: 'form'
            }), userCssClasses.searchableForm),
            input: _classnamesDefault.default(searchBoxSuit({
                descendantName: 'input'
            }), userCssClasses.searchableInput),
            submit: _classnamesDefault.default(searchBoxSuit({
                descendantName: 'submit'
            }), userCssClasses.searchableSubmit),
            submitIcon: _classnamesDefault.default(searchBoxSuit({
                descendantName: 'submitIcon'
            }), userCssClasses.searchableSubmitIcon),
            reset: _classnamesDefault.default(searchBoxSuit({
                descendantName: 'reset'
            }), userCssClasses.searchableReset),
            resetIcon: _classnamesDefault.default(searchBoxSuit({
                descendantName: 'resetIcon'
            }), userCssClasses.searchableResetIcon),
            loadingIndicator: _classnamesDefault.default(searchBoxSuit({
                descendantName: 'loadingIndicator'
            }), userCssClasses.searchableLoadingIndicator),
            loadingIcon: _classnamesDefault.default(searchBoxSuit({
                descendantName: 'loadingIcon'
            }), userCssClasses.searchableLoadingIcon)
        }
    };
    var specializedRenderer = renderer({
        containerNode: containerNode,
        cssClasses: cssClasses,
        templates: userTemplates,
        searchBoxTemplates: {
            submit: userTemplates.searchableSubmit,
            reset: userTemplates.searchableReset,
            loadingIndicator: userTemplates.searchableLoadingIndicator
        },
        renderState: {},
        searchable: searchable,
        searchablePlaceholder: searchablePlaceholder,
        searchableIsAlwaysActive: searchableIsAlwaysActive,
        showMore: showMore
    });
    var makeWidget = _connectRefinementListDefault.default(specializedRenderer, function() {
        return _preact.render(null, containerNode);
    });
    return _objectSpread(_objectSpread({}, makeWidget({
        attribute: attribute,
        operator: operator,
        limit: limit,
        showMore: showMore,
        showMoreLimit: showMoreLimit,
        sortBy: sortBy,
        escapeFacetValues: escapeFacetValues,
        transformItems: transformItems
    })), {}, {
        $$widgetType: 'ais.refinementList'
    });
};
exports.default = refinementList;

},{"preact":"26zcy","classnames":"jocGM","../../components/RefinementList/RefinementList":"2a3aK","../../connectors/refinement-list/connectRefinementList":"kkKYv","../../lib/utils":"etVYs","../../lib/suit":"du81D","../search-box/defaultTemplates":"aAfNi","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"2a3aK":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/** @jsx h */ var _preact = require("preact");
var _classnames = require("classnames");
var _classnamesDefault = parcelHelpers.interopDefault(_classnames);
var _utils = require("../../lib/utils");
var _template = require("../Template/Template");
var _templateDefault = parcelHelpers.interopDefault(_template);
var _refinementListItem = require("./RefinementListItem");
var _refinementListItemDefault = parcelHelpers.interopDefault(_refinementListItem);
var _searchBox = require("../SearchBox/SearchBox");
var _searchBoxDefault = parcelHelpers.interopDefault(_searchBox);
function _typeof(obj1) {
    if (typeof Symbol === "function" && typeof Symbol.iterator === "symbol") _typeof = function _typeof(obj) {
        return typeof obj;
    };
    else _typeof = function _typeof(obj) {
        return obj && typeof Symbol === "function" && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj;
    };
    return _typeof(obj1);
}
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        if (enumerableOnly) symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        });
        keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = arguments[i] != null ? arguments[i] : {};
        if (i % 2) ownKeys(Object(source), true).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        });
        else if (Object.getOwnPropertyDescriptors) Object.defineProperties(target, Object.getOwnPropertyDescriptors(source));
        else ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _extends() {
    _extends = Object.assign || function(target) {
        for(var i = 1; i < arguments.length; i++){
            var source = arguments[i];
            for(var key in source)if (Object.prototype.hasOwnProperty.call(source, key)) target[key] = source[key];
        }
        return target;
    };
    return _extends.apply(this, arguments);
}
function _objectWithoutProperties(source, excluded) {
    if (source == null) return {};
    var target = _objectWithoutPropertiesLoose(source, excluded);
    var key, i;
    if (Object.getOwnPropertySymbols) {
        var sourceSymbolKeys = Object.getOwnPropertySymbols(source);
        for(i = 0; i < sourceSymbolKeys.length; i++){
            key = sourceSymbolKeys[i];
            if (excluded.indexOf(key) >= 0) continue;
            if (!Object.prototype.propertyIsEnumerable.call(source, key)) continue;
            target[key] = source[key];
        }
    }
    return target;
}
function _objectWithoutPropertiesLoose(source, excluded) {
    if (source == null) return {};
    var target = {};
    var sourceKeys = Object.keys(source);
    var key, i;
    for(i = 0; i < sourceKeys.length; i++){
        key = sourceKeys[i];
        if (excluded.indexOf(key) >= 0) continue;
        target[key] = source[key];
    }
    return target;
}
function _classCallCheck(instance, Constructor) {
    if (!(instance instanceof Constructor)) throw new TypeError("Cannot call a class as a function");
}
function _defineProperties(target, props) {
    for(var i = 0; i < props.length; i++){
        var descriptor = props[i];
        descriptor.enumerable = descriptor.enumerable || false;
        descriptor.configurable = true;
        if ("value" in descriptor) descriptor.writable = true;
        Object.defineProperty(target, descriptor.key, descriptor);
    }
}
function _createClass(Constructor, protoProps, staticProps) {
    if (protoProps) _defineProperties(Constructor.prototype, protoProps);
    if (staticProps) _defineProperties(Constructor, staticProps);
    return Constructor;
}
function _inherits(subClass, superClass) {
    if (typeof superClass !== "function" && superClass !== null) throw new TypeError("Super expression must either be null or a function");
    subClass.prototype = Object.create(superClass && superClass.prototype, {
        constructor: {
            value: subClass,
            writable: true,
            configurable: true
        }
    });
    if (superClass) _setPrototypeOf(subClass, superClass);
}
function _setPrototypeOf(o1, p1) {
    _setPrototypeOf = Object.setPrototypeOf || function _setPrototypeOf(o, p) {
        o.__proto__ = p;
        return o;
    };
    return _setPrototypeOf(o1, p1);
}
function _createSuper(Derived) {
    var hasNativeReflectConstruct = _isNativeReflectConstruct();
    return function _createSuperInternal() {
        var Super = _getPrototypeOf(Derived), result;
        if (hasNativeReflectConstruct) {
            var NewTarget = _getPrototypeOf(this).constructor;
            result = Reflect.construct(Super, arguments, NewTarget);
        } else result = Super.apply(this, arguments);
        return _possibleConstructorReturn(this, result);
    };
}
function _possibleConstructorReturn(self, call) {
    if (call && (_typeof(call) === "object" || typeof call === "function")) return call;
    return _assertThisInitialized(self);
}
function _assertThisInitialized(self) {
    if (self === void 0) throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
    return self;
}
function _isNativeReflectConstruct() {
    if (typeof Reflect === "undefined" || !Reflect.construct) return false;
    if (Reflect.construct.sham) return false;
    if (typeof Proxy === "function") return true;
    try {
        Date.prototype.toString.call(Reflect.construct(Date, [], function() {}));
        return true;
    } catch (e) {
        return false;
    }
}
function _getPrototypeOf(o2) {
    _getPrototypeOf = Object.setPrototypeOf ? Object.getPrototypeOf : function _getPrototypeOf(o) {
        return o.__proto__ || Object.getPrototypeOf(o);
    };
    return _getPrototypeOf(o2);
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
var defaultProps = {
    cssClasses: {},
    depth: 0
};
function isHierarchicalMenuItem(facetValue) {
    return facetValue.data !== undefined;
}
var RefinementList = /*#__PURE__*/ function(_Component) {
    _inherits(RefinementList1, _Component);
    var _super = _createSuper(RefinementList1);
    function RefinementList1(props) {
        var _this;
        _classCallCheck(this, RefinementList1);
        _this = _super.call(this, props);
        _defineProperty(_assertThisInitialized(_this), "searchBox", _preact.createRef());
        _this.handleItemClick = _this.handleItemClick.bind(_assertThisInitialized(_this));
        return _this;
    }
    _createClass(RefinementList1, [
        {
            key: "shouldComponentUpdate",
            value: function shouldComponentUpdate(nextProps) {
                var areFacetValuesDifferent = !_utils.isEqual(this.props.facetValues, nextProps.facetValues);
                return areFacetValuesDifferent;
            }
        },
        {
            key: "refine",
            value: function refine(facetValueToRefine) {
                this.props.toggleRefinement(facetValueToRefine);
            }
        },
        {
            key: "_generateFacetItem",
            value: function _generateFacetItem(facetValue) {
                var _cx;
                var subItems;
                if (isHierarchicalMenuItem(facetValue) && Array.isArray(facetValue.data) && facetValue.data.length > 0) {
                    var _this$props$cssClasse = this.props.cssClasses, root = _this$props$cssClasse.root, cssClasses = _objectWithoutProperties(_this$props$cssClasse, [
                        "root"
                    ]);
                    subItems = _preact.h(RefinementList1, _extends({}, this.props, {
                        cssClasses: cssClasses,
                        depth: this.props.depth + 1,
                        facetValues: facetValue.data,
                        showMore: false,
                        className: this.props.cssClasses.childList
                    }));
                }
                var url = this.props.createURL(facetValue.value);
                var templateData = _objectSpread(_objectSpread({}, facetValue), {}, {
                    url: url,
                    attribute: this.props.attribute,
                    cssClasses: this.props.cssClasses,
                    isFromSearch: this.props.isFromSearch
                });
                var key = facetValue.value;
                if (facetValue.isRefined !== undefined) key += "/".concat(facetValue.isRefined);
                if (facetValue.count !== undefined) key += "/".concat(facetValue.count);
                var refinementListItemClassName = _classnamesDefault.default(this.props.cssClasses.item, (_cx = {}, _defineProperty(_cx, this.props.cssClasses.selectedItem, facetValue.isRefined), _defineProperty(_cx, this.props.cssClasses.disabledItem, !facetValue.count), _defineProperty(_cx, this.props.cssClasses.parentItem, isHierarchicalMenuItem(facetValue) && Array.isArray(facetValue.data) && facetValue.data.length > 0), _cx));
                return _preact.h(_refinementListItemDefault.default, {
                    templateKey: "item",
                    key: key,
                    facetValueToRefine: facetValue.value,
                    handleClick: this.handleItemClick,
                    isRefined: facetValue.isRefined,
                    className: refinementListItemClassName,
                    subItems: subItems,
                    templateData: templateData,
                    templateProps: this.props.templateProps
                });
            } // Click events on DOM tree like LABEL > INPUT will result in two click events
        },
        {
            key: "handleItemClick",
            value: function handleItemClick(_ref) {
                var facetValueToRefine = _ref.facetValueToRefine, isRefined = _ref.isRefined, originalEvent = _ref.originalEvent;
                if (_utils.isSpecialClick(originalEvent)) // do not alter the default browser behavior
                // if one special key is down
                return;
                if (!(originalEvent.target instanceof HTMLElement) || !(originalEvent.target.parentNode instanceof HTMLElement)) return;
                if (isRefined && originalEvent.target.parentNode.querySelector('input[type="radio"]:checked')) // Prevent refinement for being reset if the user clicks on an already checked radio button
                return;
                if (originalEvent.target.tagName === 'INPUT') {
                    this.refine(facetValueToRefine);
                    return;
                }
                var parent = originalEvent.target;
                while(parent !== originalEvent.currentTarget){
                    if (parent.tagName === 'LABEL' && (parent.querySelector('input[type="checkbox"]') || parent.querySelector('input[type="radio"]'))) return;
                    if (parent.tagName === 'A' && parent.href) originalEvent.preventDefault();
                    parent = parent.parentNode;
                }
                originalEvent.stopPropagation();
                this.refine(facetValueToRefine);
            }
        },
        {
            key: "componentWillReceiveProps",
            value: function componentWillReceiveProps(nextProps) {
                if (this.searchBox.current && !nextProps.isFromSearch) this.searchBox.current.resetInput();
            }
        },
        {
            key: "refineFirstValue",
            value: function refineFirstValue() {
                var firstValue = this.props.facetValues && this.props.facetValues[0];
                if (firstValue) {
                    var actualValue = firstValue.value;
                    this.props.toggleRefinement(actualValue);
                }
            }
        },
        {
            key: "render",
            value: function render() {
                var _this2 = this;
                var showMoreButtonClassName = _classnamesDefault.default(this.props.cssClasses.showMore, _defineProperty({}, this.props.cssClasses.disabledShowMore, !(this.props.showMore === true && this.props.canToggleShowMore)));
                var showMoreButton = this.props.showMore === true && _preact.h(_templateDefault.default, _extends({}, this.props.templateProps, {
                    templateKey: "showMoreText",
                    rootTagName: "button",
                    rootProps: {
                        className: showMoreButtonClassName,
                        disabled: !this.props.canToggleShowMore,
                        onClick: this.props.toggleShowMore
                    },
                    data: {
                        isShowingMore: this.props.isShowingMore
                    }
                }));
                var shouldDisableSearchBox = this.props.searchIsAlwaysActive !== true && !(this.props.isFromSearch || !this.props.hasExhaustiveItems);
                var templates = this.props.searchBoxTemplateProps ? this.props.searchBoxTemplateProps.templates : undefined;
                var searchBox = this.props.searchFacetValues && _preact.h("div", {
                    className: this.props.cssClasses.searchBox
                }, _preact.h(_searchBoxDefault.default, {
                    ref: this.searchBox,
                    placeholder: this.props.searchPlaceholder,
                    disabled: shouldDisableSearchBox,
                    cssClasses: this.props.cssClasses.searchable,
                    templates: templates,
                    onChange: function onChange(event) {
                        return _this2.props.searchFacetValues(event.target.value);
                    },
                    onReset: function onReset() {
                        return _this2.props.searchFacetValues('');
                    },
                    onSubmit: function onSubmit() {
                        return _this2.refineFirstValue();
                    } // This sets the search box to a controlled state because
                    ,
                    searchAsYouType: false
                }));
                var facetValues = this.props.facetValues && this.props.facetValues.length > 0 && _preact.h("ul", {
                    className: this.props.cssClasses.list
                }, this.props.facetValues.map(this._generateFacetItem, this));
                var noResults = this.props.searchFacetValues && this.props.isFromSearch && (!this.props.facetValues || this.props.facetValues.length === 0) && _preact.h(_templateDefault.default, _extends({}, this.props.templateProps, {
                    templateKey: "searchableNoResults",
                    rootProps: {
                        className: this.props.cssClasses.noResults
                    }
                }));
                var rootClassName = _classnamesDefault.default(this.props.cssClasses.root, _defineProperty({}, this.props.cssClasses.noRefinementRoot, !this.props.facetValues || this.props.facetValues.length === 0), this.props.className);
                return _preact.h("div", {
                    className: rootClassName
                }, this.props.children, searchBox, facetValues, noResults, showMoreButton);
            }
        }
    ]);
    return RefinementList1;
}(_preact.Component);
_defineProperty(RefinementList, "defaultProps", defaultProps);
exports.default = RefinementList;

},{"preact":"26zcy","classnames":"jocGM","../../lib/utils":"etVYs","../Template/Template":"aVPg5","./RefinementListItem":"iBH7m","../SearchBox/SearchBox":"acfnu","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"iBH7m":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/** @jsx h */ var _preact = require("preact");
var _template = require("../Template/Template");
var _templateDefault = parcelHelpers.interopDefault(_template);
function _extends() {
    _extends = Object.assign || function(target) {
        for(var i = 1; i < arguments.length; i++){
            var source = arguments[i];
            for(var key in source)if (Object.prototype.hasOwnProperty.call(source, key)) target[key] = source[key];
        }
        return target;
    };
    return _extends.apply(this, arguments);
}
function RefinementListItem(_ref) {
    var className = _ref.className, handleClick = _ref.handleClick, facetValueToRefine = _ref.facetValueToRefine, isRefined = _ref.isRefined, templateProps = _ref.templateProps, templateKey = _ref.templateKey, templateData = _ref.templateData, subItems = _ref.subItems;
    return _preact.h("li", {
        className: className,
        onClick: function onClick(originalEvent) {
            handleClick({
                facetValueToRefine: facetValueToRefine,
                isRefined: isRefined,
                originalEvent: originalEvent
            });
        }
    }, _preact.h(_templateDefault.default, _extends({}, templateProps, {
        templateKey: templateKey,
        data: templateData
    })), subItems);
}
exports.default = RefinementListItem;

},{"preact":"26zcy","../Template/Template":"aVPg5","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"acfnu":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/** @jsx h */ var _preact = require("preact");
var _utils = require("../../lib/utils");
var _template = require("../Template/Template");
var _templateDefault = parcelHelpers.interopDefault(_template);
function _typeof(obj1) {
    if (typeof Symbol === "function" && typeof Symbol.iterator === "symbol") _typeof = function _typeof(obj) {
        return typeof obj;
    };
    else _typeof = function _typeof(obj) {
        return obj && typeof Symbol === "function" && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj;
    };
    return _typeof(obj1);
}
function _classCallCheck(instance, Constructor) {
    if (!(instance instanceof Constructor)) throw new TypeError("Cannot call a class as a function");
}
function _defineProperties(target, props) {
    for(var i = 0; i < props.length; i++){
        var descriptor = props[i];
        descriptor.enumerable = descriptor.enumerable || false;
        descriptor.configurable = true;
        if ("value" in descriptor) descriptor.writable = true;
        Object.defineProperty(target, descriptor.key, descriptor);
    }
}
function _createClass(Constructor, protoProps, staticProps) {
    if (protoProps) _defineProperties(Constructor.prototype, protoProps);
    if (staticProps) _defineProperties(Constructor, staticProps);
    return Constructor;
}
function _inherits(subClass, superClass) {
    if (typeof superClass !== "function" && superClass !== null) throw new TypeError("Super expression must either be null or a function");
    subClass.prototype = Object.create(superClass && superClass.prototype, {
        constructor: {
            value: subClass,
            writable: true,
            configurable: true
        }
    });
    if (superClass) _setPrototypeOf(subClass, superClass);
}
function _setPrototypeOf(o1, p1) {
    _setPrototypeOf = Object.setPrototypeOf || function _setPrototypeOf(o, p) {
        o.__proto__ = p;
        return o;
    };
    return _setPrototypeOf(o1, p1);
}
function _createSuper(Derived) {
    var hasNativeReflectConstruct = _isNativeReflectConstruct();
    return function _createSuperInternal() {
        var Super = _getPrototypeOf(Derived), result;
        if (hasNativeReflectConstruct) {
            var NewTarget = _getPrototypeOf(this).constructor;
            result = Reflect.construct(Super, arguments, NewTarget);
        } else result = Super.apply(this, arguments);
        return _possibleConstructorReturn(this, result);
    };
}
function _possibleConstructorReturn(self, call) {
    if (call && (_typeof(call) === "object" || typeof call === "function")) return call;
    return _assertThisInitialized(self);
}
function _assertThisInitialized(self) {
    if (self === void 0) throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
    return self;
}
function _isNativeReflectConstruct() {
    if (typeof Reflect === "undefined" || !Reflect.construct) return false;
    if (Reflect.construct.sham) return false;
    if (typeof Proxy === "function") return true;
    try {
        Date.prototype.toString.call(Reflect.construct(Date, [], function() {}));
        return true;
    } catch (e) {
        return false;
    }
}
function _getPrototypeOf(o2) {
    _getPrototypeOf = Object.setPrototypeOf ? Object.getPrototypeOf : function _getPrototypeOf(o) {
        return o.__proto__ || Object.getPrototypeOf(o);
    };
    return _getPrototypeOf(o2);
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
var defaultProps = {
    query: '',
    showSubmit: true,
    showReset: true,
    showLoadingIndicator: true,
    autofocus: false,
    searchAsYouType: true,
    isSearchStalled: false,
    disabled: false,
    onChange: _utils.noop,
    onSubmit: _utils.noop,
    onReset: _utils.noop,
    refine: _utils.noop
};
var SearchBox = /*#__PURE__*/ function(_Component) {
    _inherits(SearchBox1, _Component);
    var _super = _createSuper(SearchBox1);
    function SearchBox1() {
        var _this;
        _classCallCheck(this, SearchBox1);
        for(var _len = arguments.length, args = new Array(_len), _key = 0; _key < _len; _key++)args[_key] = arguments[_key];
        _this = _super.call.apply(_super, [
            this
        ].concat(args));
        _defineProperty(_assertThisInitialized(_this), "state", {
            query: _this.props.query,
            focused: false
        });
        _defineProperty(_assertThisInitialized(_this), "input", _preact.createRef());
        _defineProperty(_assertThisInitialized(_this), "onInput", function(event) {
            var _this$props = _this.props, searchAsYouType = _this$props.searchAsYouType, refine = _this$props.refine, onChange = _this$props.onChange;
            var query = event.target.value;
            if (searchAsYouType) refine(query);
            _this.setState({
                query: query
            });
            onChange(event);
        });
        _defineProperty(_assertThisInitialized(_this), "onSubmit", function(event) {
            var _this$props2 = _this.props, searchAsYouType = _this$props2.searchAsYouType, refine = _this$props2.refine, onSubmit = _this$props2.onSubmit;
            event.preventDefault();
            event.stopPropagation();
            if (_this.input.current) _this.input.current.blur();
            if (!searchAsYouType) refine(_this.state.query);
            onSubmit(event);
            return false;
        });
        _defineProperty(_assertThisInitialized(_this), "onReset", function(event) {
            var _this$props3 = _this.props, refine = _this$props3.refine, onReset = _this$props3.onReset;
            var query = '';
            if (_this.input.current) _this.input.current.focus();
            refine(query);
            _this.setState({
                query: query
            });
            onReset(event);
        });
        _defineProperty(_assertThisInitialized(_this), "onBlur", function() {
            _this.setState({
                focused: false
            });
        });
        _defineProperty(_assertThisInitialized(_this), "onFocus", function() {
            _this.setState({
                focused: true
            });
        });
        return _this;
    }
    _createClass(SearchBox1, [
        {
            key: "resetInput",
            value: /**
     * This public method is used in the RefinementList SFFV search box
     * to reset the input state when an item is selected.
     *
     * @see RefinementList#componentWillReceiveProps
     * @return {undefined}
     */ function resetInput() {
                this.setState({
                    query: ''
                });
            }
        },
        {
            key: "componentWillReceiveProps",
            value: function componentWillReceiveProps(nextProps) {
                /**
       * when the user is typing, we don't want to replace the query typed
       * by the user (state.query) with the query exposed by the connector (props.query)
       * see: https://github.com/algolia/instantsearch.js/issues/4141
       */ if (!this.state.focused && nextProps.query !== this.state.query) this.setState({
                    query: nextProps.query
                });
            }
        },
        {
            key: "render",
            value: function render() {
                var _this$props4 = this.props, cssClasses = _this$props4.cssClasses, placeholder = _this$props4.placeholder, autofocus = _this$props4.autofocus, showSubmit = _this$props4.showSubmit, showReset = _this$props4.showReset, showLoadingIndicator = _this$props4.showLoadingIndicator, templates = _this$props4.templates, isSearchStalled = _this$props4.isSearchStalled;
                return _preact.h("div", {
                    className: cssClasses.root
                }, _preact.h("form", {
                    action: "",
                    role: "search",
                    className: cssClasses.form,
                    noValidate: true,
                    onSubmit: this.onSubmit // @ts-expect-error `onReset` attibute is missing in preact 10.0.0 JSX types
                    ,
                    onReset: this.onReset
                }, _preact.h("input", {
                    ref: this.input,
                    value: this.state.query,
                    disabled: this.props.disabled,
                    className: cssClasses.input,
                    type: "search",
                    placeholder: placeholder,
                    autoFocus: autofocus,
                    autoComplete: "off",
                    autoCorrect: "off" // @ts-expect-error `autoCapitalize` attibute is missing in preact 10.0.0 JSX types
                    ,
                    autoCapitalize: "off",
                    spellCheck: "false",
                    maxLength: 512,
                    onInput: this.onInput,
                    onBlur: this.onBlur,
                    onFocus: this.onFocus
                }), _preact.h(_templateDefault.default, {
                    templateKey: "submit",
                    rootTagName: "button",
                    rootProps: {
                        className: cssClasses.submit,
                        type: 'submit',
                        title: 'Submit the search query.',
                        hidden: !showSubmit
                    },
                    templates: templates,
                    data: {
                        cssClasses: cssClasses
                    }
                }), _preact.h(_templateDefault.default, {
                    templateKey: "reset",
                    rootTagName: "button",
                    rootProps: {
                        className: cssClasses.reset,
                        type: 'reset',
                        title: 'Clear the search query.',
                        hidden: !(showReset && this.state.query.trim() && !isSearchStalled)
                    },
                    templates: templates,
                    data: {
                        cssClasses: cssClasses
                    }
                }), showLoadingIndicator && _preact.h(_templateDefault.default, {
                    templateKey: "loadingIndicator",
                    rootTagName: "span",
                    rootProps: {
                        className: cssClasses.loadingIndicator,
                        hidden: !isSearchStalled
                    },
                    templates: templates,
                    data: {
                        cssClasses: cssClasses
                    }
                })));
            }
        }
    ]);
    return SearchBox1;
}(_preact.Component);
_defineProperty(SearchBox, "defaultProps", defaultProps);
exports.default = SearchBox;

},{"preact":"26zcy","../../lib/utils":"etVYs","../Template/Template":"aVPg5","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"aAfNi":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/* eslint max-len: 0 */ exports.default = {
    reset: "\n<svg class=\"{{cssClasses.resetIcon}}\" xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 20 20\" width=\"10\" height=\"10\">\n  <path d=\"M8.114 10L.944 2.83 0 1.885 1.886 0l.943.943L10 8.113l7.17-7.17.944-.943L20 1.886l-.943.943-7.17 7.17 7.17 7.17.943.944L18.114 20l-.943-.943-7.17-7.17-7.17 7.17-.944.943L0 18.114l.943-.943L8.113 10z\"></path>\n</svg>\n  ",
    submit: "\n<svg class=\"{{cssClasses.submitIcon}}\" xmlns=\"http://www.w3.org/2000/svg\" width=\"10\" height=\"10\" viewBox=\"0 0 40 40\">\n  <path d=\"M26.804 29.01c-2.832 2.34-6.465 3.746-10.426 3.746C7.333 32.756 0 25.424 0 16.378 0 7.333 7.333 0 16.378 0c9.046 0 16.378 7.333 16.378 16.378 0 3.96-1.406 7.594-3.746 10.426l10.534 10.534c.607.607.61 1.59-.004 2.202-.61.61-1.597.61-2.202.004L26.804 29.01zm-10.426.627c7.323 0 13.26-5.936 13.26-13.26 0-7.32-5.937-13.257-13.26-13.257C9.056 3.12 3.12 9.056 3.12 16.378c0 7.323 5.936 13.26 13.258 13.26z\"></path>\n</svg>\n  ",
    loadingIndicator: "\n<svg class=\"{{cssClasses.loadingIcon}}\" width=\"16\" height=\"16\" viewBox=\"0 0 38 38\" xmlns=\"http://www.w3.org/2000/svg\" stroke=\"#444\">\n  <g fill=\"none\" fillRule=\"evenodd\">\n    <g transform=\"translate(1 1)\" strokeWidth=\"2\">\n      <circle strokeOpacity=\".5\" cx=\"18\" cy=\"18\" r=\"18\" />\n      <path d=\"M36 18c0-9.94-8.06-18-18-18\">\n        <animateTransform\n          attributeName=\"transform\"\n          type=\"rotate\"\n          from=\"0 18 18\"\n          to=\"360 18 18\"\n          dur=\"1s\"\n          repeatCount=\"indefinite\"\n        />\n      </path>\n    </g>\n  </g>\n</svg>\n  "
};

},{"@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"aGC8J":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/** @jsx h */ var _preact = require("preact");
var _classnames = require("classnames");
var _classnamesDefault = parcelHelpers.interopDefault(_classnames);
var _pagination = require("../../components/Pagination/Pagination");
var _paginationDefault = parcelHelpers.interopDefault(_pagination);
var _connectPagination = require("../../connectors/pagination/connectPagination");
var _connectPaginationDefault = parcelHelpers.interopDefault(_connectPagination);
var _utils = require("../../lib/utils");
var _suit = require("../../lib/suit");
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        if (enumerableOnly) symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        });
        keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = arguments[i] != null ? arguments[i] : {};
        if (i % 2) ownKeys(Object(source), true).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        });
        else if (Object.getOwnPropertyDescriptors) Object.defineProperties(target, Object.getOwnPropertyDescriptors(source));
        else ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
var withUsage = _utils.createDocumentationMessageGenerator({
    name: 'pagination'
});
var suit = _suit.component('Pagination');
var defaultTemplates = {
    previous: '‹',
    next: '›',
    first: '«',
    last: '»'
};
var renderer = function renderer(_ref) {
    var containerNode = _ref.containerNode, cssClasses = _ref.cssClasses, templates = _ref.templates, totalPages = _ref.totalPages, showFirst = _ref.showFirst, showLast = _ref.showLast, showPrevious = _ref.showPrevious, showNext = _ref.showNext, scrollToNode = _ref.scrollToNode;
    return function(_ref2, isFirstRendering) {
        var createURL = _ref2.createURL, currentRefinement = _ref2.currentRefinement, nbHits = _ref2.nbHits, nbPages = _ref2.nbPages, pages = _ref2.pages, isFirstPage = _ref2.isFirstPage, isLastPage = _ref2.isLastPage, refine = _ref2.refine;
        if (isFirstRendering) return;
        var setCurrentPage = function setCurrentPage(pageNumber) {
            refine(pageNumber);
            if (scrollToNode !== false) scrollToNode.scrollIntoView();
        };
        _preact.render(_preact.h(_paginationDefault.default, {
            createURL: createURL,
            cssClasses: cssClasses,
            currentPage: currentRefinement,
            templates: templates,
            nbHits: nbHits,
            nbPages: nbPages,
            pages: pages,
            totalPages: totalPages,
            isFirstPage: isFirstPage,
            isLastPage: isLastPage,
            setCurrentPage: setCurrentPage,
            showFirst: showFirst,
            showLast: showLast,
            showPrevious: showPrevious,
            showNext: showNext
        }), containerNode);
    };
};
function pagination(widgetParams) {
    var _ref3 = widgetParams || {}, container = _ref3.container, _ref3$templates = _ref3.templates, userTemplates = _ref3$templates === void 0 ? {} : _ref3$templates, _ref3$cssClasses = _ref3.cssClasses, userCssClasses = _ref3$cssClasses === void 0 ? {} : _ref3$cssClasses, totalPages = _ref3.totalPages, padding = _ref3.padding, _ref3$showFirst = _ref3.showFirst, showFirst = _ref3$showFirst === void 0 ? true : _ref3$showFirst, _ref3$showLast = _ref3.showLast, showLast = _ref3$showLast === void 0 ? true : _ref3$showLast, _ref3$showPrevious = _ref3.showPrevious, showPrevious = _ref3$showPrevious === void 0 ? true : _ref3$showPrevious, _ref3$showNext = _ref3.showNext, showNext = _ref3$showNext === void 0 ? true : _ref3$showNext, _ref3$scrollTo = _ref3.scrollTo, userScrollTo = _ref3$scrollTo === void 0 ? 'body' : _ref3$scrollTo;
    if (!container) throw new Error(withUsage('The `container` option is required.'));
    var containerNode = _utils.getContainerNode(container);
    var scrollTo = userScrollTo === true ? 'body' : userScrollTo;
    var scrollToNode = scrollTo !== false ? _utils.getContainerNode(scrollTo) : false;
    var cssClasses = {
        root: _classnamesDefault.default(suit(), userCssClasses.root),
        noRefinementRoot: _classnamesDefault.default(suit({
            modifierName: 'noRefinement'
        }), userCssClasses.noRefinementRoot),
        list: _classnamesDefault.default(suit({
            descendantName: 'list'
        }), userCssClasses.list),
        item: _classnamesDefault.default(suit({
            descendantName: 'item'
        }), userCssClasses.item),
        firstPageItem: _classnamesDefault.default(suit({
            descendantName: 'item',
            modifierName: 'firstPage'
        }), userCssClasses.firstPageItem),
        lastPageItem: _classnamesDefault.default(suit({
            descendantName: 'item',
            modifierName: 'lastPage'
        }), userCssClasses.lastPageItem),
        previousPageItem: _classnamesDefault.default(suit({
            descendantName: 'item',
            modifierName: 'previousPage'
        }), userCssClasses.previousPageItem),
        nextPageItem: _classnamesDefault.default(suit({
            descendantName: 'item',
            modifierName: 'nextPage'
        }), userCssClasses.nextPageItem),
        pageItem: _classnamesDefault.default(suit({
            descendantName: 'item',
            modifierName: 'page'
        }), userCssClasses.pageItem),
        selectedItem: _classnamesDefault.default(suit({
            descendantName: 'item',
            modifierName: 'selected'
        }), userCssClasses.selectedItem),
        disabledItem: _classnamesDefault.default(suit({
            descendantName: 'item',
            modifierName: 'disabled'
        }), userCssClasses.disabledItem),
        link: _classnamesDefault.default(suit({
            descendantName: 'link'
        }), userCssClasses.link)
    };
    var templates = _objectSpread(_objectSpread({}, defaultTemplates), userTemplates);
    var specializedRenderer = renderer({
        containerNode: containerNode,
        cssClasses: cssClasses,
        templates: templates,
        showFirst: showFirst,
        showLast: showLast,
        showPrevious: showPrevious,
        showNext: showNext,
        padding: padding,
        scrollToNode: scrollToNode
    });
    var makeWidget = _connectPaginationDefault.default(specializedRenderer, function() {
        return _preact.render(null, containerNode);
    });
    return _objectSpread(_objectSpread({}, makeWidget({
        totalPages: totalPages,
        padding: padding
    })), {}, {
        $$widgetType: 'ais.pagination'
    });
}
exports.default = pagination;

},{"preact":"26zcy","classnames":"jocGM","../../components/Pagination/Pagination":"4kiOm","../../connectors/pagination/connectPagination":"bHouJ","../../lib/utils":"etVYs","../../lib/suit":"du81D","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"4kiOm":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/** @jsx h */ var _preact = require("preact");
var _classnames = require("classnames");
var _classnamesDefault = parcelHelpers.interopDefault(_classnames);
var _paginationLink = require("./PaginationLink");
var _paginationLinkDefault = parcelHelpers.interopDefault(_paginationLink);
var _utils = require("../../lib/utils");
function _typeof(obj1) {
    if (typeof Symbol === "function" && typeof Symbol.iterator === "symbol") _typeof = function _typeof(obj) {
        return typeof obj;
    };
    else _typeof = function _typeof(obj) {
        return obj && typeof Symbol === "function" && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj;
    };
    return _typeof(obj1);
}
function _classCallCheck(instance, Constructor) {
    if (!(instance instanceof Constructor)) throw new TypeError("Cannot call a class as a function");
}
function _defineProperties(target, props) {
    for(var i = 0; i < props.length; i++){
        var descriptor = props[i];
        descriptor.enumerable = descriptor.enumerable || false;
        descriptor.configurable = true;
        if ("value" in descriptor) descriptor.writable = true;
        Object.defineProperty(target, descriptor.key, descriptor);
    }
}
function _createClass(Constructor, protoProps, staticProps) {
    if (protoProps) _defineProperties(Constructor.prototype, protoProps);
    if (staticProps) _defineProperties(Constructor, staticProps);
    return Constructor;
}
function _inherits(subClass, superClass) {
    if (typeof superClass !== "function" && superClass !== null) throw new TypeError("Super expression must either be null or a function");
    subClass.prototype = Object.create(superClass && superClass.prototype, {
        constructor: {
            value: subClass,
            writable: true,
            configurable: true
        }
    });
    if (superClass) _setPrototypeOf(subClass, superClass);
}
function _setPrototypeOf(o1, p1) {
    _setPrototypeOf = Object.setPrototypeOf || function _setPrototypeOf(o, p) {
        o.__proto__ = p;
        return o;
    };
    return _setPrototypeOf(o1, p1);
}
function _createSuper(Derived) {
    var hasNativeReflectConstruct = _isNativeReflectConstruct();
    return function _createSuperInternal() {
        var Super = _getPrototypeOf(Derived), result;
        if (hasNativeReflectConstruct) {
            var NewTarget = _getPrototypeOf(this).constructor;
            result = Reflect.construct(Super, arguments, NewTarget);
        } else result = Super.apply(this, arguments);
        return _possibleConstructorReturn(this, result);
    };
}
function _possibleConstructorReturn(self, call) {
    if (call && (_typeof(call) === "object" || typeof call === "function")) return call;
    return _assertThisInitialized(self);
}
function _assertThisInitialized(self) {
    if (self === void 0) throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
    return self;
}
function _isNativeReflectConstruct() {
    if (typeof Reflect === "undefined" || !Reflect.construct) return false;
    if (Reflect.construct.sham) return false;
    if (typeof Proxy === "function") return true;
    try {
        Date.prototype.toString.call(Reflect.construct(Date, [], function() {}));
        return true;
    } catch (e) {
        return false;
    }
}
function _getPrototypeOf(o2) {
    _getPrototypeOf = Object.setPrototypeOf ? Object.getPrototypeOf : function _getPrototypeOf(o) {
        return o.__proto__ || Object.getPrototypeOf(o);
    };
    return _getPrototypeOf(o2);
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
var Pagination = /*#__PURE__*/ function(_Component) {
    _inherits(Pagination1, _Component);
    var _super = _createSuper(Pagination1);
    function Pagination1() {
        var _this;
        _classCallCheck(this, Pagination1);
        for(var _len = arguments.length, args = new Array(_len), _key = 0; _key < _len; _key++)args[_key] = arguments[_key];
        _this = _super.call.apply(_super, [
            this
        ].concat(args));
        _defineProperty(_assertThisInitialized(_this), "handleClick", function(pageNumber, event) {
            if (_utils.isSpecialClick(event)) // do not alter the default browser behavior
            // if one special key is down
            return;
            event.preventDefault();
            _this.props.setCurrentPage(pageNumber);
        });
        return _this;
    }
    _createClass(Pagination1, [
        {
            key: "pageLink",
            value: function pageLink(_ref) {
                var label = _ref.label, ariaLabel = _ref.ariaLabel, pageNumber = _ref.pageNumber, _ref$additionalClassN = _ref.additionalClassName, additionalClassName = _ref$additionalClassN === void 0 ? null : _ref$additionalClassN, _ref$isDisabled = _ref.isDisabled, isDisabled = _ref$isDisabled === void 0 ? false : _ref$isDisabled, _ref$isSelected = _ref.isSelected, isSelected = _ref$isSelected === void 0 ? false : _ref$isSelected, createURL = _ref.createURL;
                var cssClasses = {
                    item: _classnamesDefault.default(this.props.cssClasses.item, additionalClassName),
                    link: this.props.cssClasses.link
                };
                if (isDisabled) cssClasses.item = _classnamesDefault.default(cssClasses.item, this.props.cssClasses.disabledItem);
                else if (isSelected) cssClasses.item = _classnamesDefault.default(cssClasses.item, this.props.cssClasses.selectedItem);
                var url = createURL && !isDisabled ? createURL(pageNumber) : '#';
                return _preact.h(_paginationLinkDefault.default, {
                    ariaLabel: ariaLabel,
                    cssClasses: cssClasses,
                    handleClick: this.handleClick,
                    isDisabled: isDisabled,
                    key: label + pageNumber + ariaLabel,
                    label: label,
                    pageNumber: pageNumber,
                    url: url
                });
            }
        },
        {
            key: "previousPageLink",
            value: function previousPageLink(_ref2) {
                var isFirstPage = _ref2.isFirstPage, currentPage = _ref2.currentPage, createURL = _ref2.createURL;
                return this.pageLink({
                    ariaLabel: 'Previous',
                    additionalClassName: this.props.cssClasses.previousPageItem,
                    isDisabled: isFirstPage,
                    label: this.props.templates.previous,
                    pageNumber: currentPage - 1,
                    createURL: createURL
                });
            }
        },
        {
            key: "nextPageLink",
            value: function nextPageLink(_ref3) {
                var isLastPage = _ref3.isLastPage, currentPage = _ref3.currentPage, createURL = _ref3.createURL;
                return this.pageLink({
                    ariaLabel: 'Next',
                    additionalClassName: this.props.cssClasses.nextPageItem,
                    isDisabled: isLastPage,
                    label: this.props.templates.next,
                    pageNumber: currentPage + 1,
                    createURL: createURL
                });
            }
        },
        {
            key: "firstPageLink",
            value: function firstPageLink(_ref4) {
                var isFirstPage = _ref4.isFirstPage, createURL = _ref4.createURL;
                return this.pageLink({
                    ariaLabel: 'First',
                    additionalClassName: this.props.cssClasses.firstPageItem,
                    isDisabled: isFirstPage,
                    label: this.props.templates.first,
                    pageNumber: 0,
                    createURL: createURL
                });
            }
        },
        {
            key: "lastPageLink",
            value: function lastPageLink(_ref5) {
                var isLastPage = _ref5.isLastPage, nbPages = _ref5.nbPages, createURL = _ref5.createURL;
                return this.pageLink({
                    ariaLabel: 'Last',
                    additionalClassName: this.props.cssClasses.lastPageItem,
                    isDisabled: isLastPage,
                    label: this.props.templates.last,
                    pageNumber: nbPages - 1,
                    createURL: createURL
                });
            }
        },
        {
            key: "pages",
            value: function pages(_ref6) {
                var _this2 = this;
                var currentPage = _ref6.currentPage, _pages = _ref6.pages, createURL = _ref6.createURL;
                return _pages.map(function(pageNumber) {
                    return _this2.pageLink({
                        ariaLabel: pageNumber + 1,
                        additionalClassName: _this2.props.cssClasses.pageItem,
                        isSelected: pageNumber === currentPage,
                        label: pageNumber + 1,
                        pageNumber: pageNumber,
                        createURL: createURL
                    });
                });
            }
        },
        {
            key: "render",
            value: function render() {
                return _preact.h("div", {
                    className: _classnamesDefault.default(this.props.cssClasses.root, _defineProperty({}, this.props.cssClasses.noRefinementRoot, this.props.nbPages <= 1))
                }, _preact.h("ul", {
                    className: this.props.cssClasses.list
                }, this.props.showFirst && this.firstPageLink(this.props), this.props.showPrevious && this.previousPageLink(this.props), this.pages(this.props), this.props.showNext && this.nextPageLink(this.props), this.props.showLast && this.lastPageLink(this.props)));
            }
        }
    ]);
    return Pagination1;
}(_preact.Component);
Pagination.defaultProps = {
    nbHits: 0,
    currentPage: 0,
    nbPages: 0
};
exports.default = Pagination;

},{"preact":"26zcy","classnames":"jocGM","./PaginationLink":"7xJp0","../../lib/utils":"etVYs","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"7xJp0":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/** @jsx h */ var _preact = require("preact");
function PaginationLink(_ref) {
    var cssClasses = _ref.cssClasses, label = _ref.label, ariaLabel = _ref.ariaLabel, url = _ref.url, isDisabled = _ref.isDisabled, handleClick = _ref.handleClick, pageNumber = _ref.pageNumber;
    if (isDisabled) return _preact.h("li", {
        className: cssClasses.item
    }, _preact.h("span", {
        className: cssClasses.link,
        dangerouslySetInnerHTML: {
            __html: label
        }
    }));
    return _preact.h("li", {
        className: cssClasses.item
    }, _preact.h("a", {
        className: cssClasses.link,
        "aria-label": ariaLabel,
        href: url,
        onClick: function onClick(event) {
            return handleClick(pageNumber, event);
        },
        dangerouslySetInnerHTML: {
            __html: label
        }
    }));
}
exports.default = PaginationLink;

},{"preact":"26zcy","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"4PvhT":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/** @jsx h */ var _preact = require("preact");
var _classnames = require("classnames");
var _classnamesDefault = parcelHelpers.interopDefault(_classnames);
var _slider = require("../../components/Slider/Slider");
var _sliderDefault = parcelHelpers.interopDefault(_slider);
var _connectRange = require("../../connectors/range/connectRange");
var _connectRangeDefault = parcelHelpers.interopDefault(_connectRange);
var _utils = require("../../lib/utils");
var _suit = require("../../lib/suit");
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        if (enumerableOnly) symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        });
        keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = arguments[i] != null ? arguments[i] : {};
        if (i % 2) ownKeys(Object(source), true).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        });
        else if (Object.getOwnPropertyDescriptors) Object.defineProperties(target, Object.getOwnPropertyDescriptors(source));
        else ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
function _slicedToArray(arr, i) {
    return _arrayWithHoles(arr) || _iterableToArrayLimit(arr, i) || _unsupportedIterableToArray(arr, i) || _nonIterableRest();
}
function _nonIterableRest() {
    throw new TypeError("Invalid attempt to destructure non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
}
function _unsupportedIterableToArray(o, minLen) {
    if (!o) return;
    if (typeof o === "string") return _arrayLikeToArray(o, minLen);
    var n = Object.prototype.toString.call(o).slice(8, -1);
    if (n === "Object" && o.constructor) n = o.constructor.name;
    if (n === "Map" || n === "Set") return Array.from(o);
    if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen);
}
function _arrayLikeToArray(arr, len) {
    if (len == null || len > arr.length) len = arr.length;
    for(var i = 0, arr2 = new Array(len); i < len; i++)arr2[i] = arr[i];
    return arr2;
}
function _iterableToArrayLimit(arr, i) {
    if (typeof Symbol === "undefined" || !(Symbol.iterator in Object(arr))) return;
    var _arr = [];
    var _n = true;
    var _d = false;
    var _e = undefined;
    try {
        for(var _i = arr[Symbol.iterator](), _s; !(_n = (_s = _i.next()).done); _n = true){
            _arr.push(_s.value);
            if (i && _arr.length === i) break;
        }
    } catch (err) {
        _d = true;
        _e = err;
    } finally{
        try {
            if (!_n && _i["return"] != null) _i["return"]();
        } finally{
            if (_d) throw _e;
        }
    }
    return _arr;
}
function _arrayWithHoles(arr) {
    if (Array.isArray(arr)) return arr;
}
var withUsage = _utils.createDocumentationMessageGenerator({
    name: 'range-slider'
});
var suit = _suit.component('RangeSlider');
var renderer = function renderer(_ref) {
    var containerNode = _ref.containerNode, cssClasses = _ref.cssClasses, pips = _ref.pips, step = _ref.step, tooltips = _ref.tooltips;
    return function(_ref2, isFirstRendering) {
        var refine = _ref2.refine, range = _ref2.range, start = _ref2.start;
        if (isFirstRendering) // There's no information at this point, let's render nothing.
        return;
        var minRange = range.min, maxRange = range.max;
        var _start = _slicedToArray(start, 2), minStart = _start[0], maxStart = _start[1];
        var minFinite = minStart === -Infinity ? minRange : minStart;
        var maxFinite = maxStart === Infinity ? maxRange : maxStart; // Clamp values to the range for avoid extra rendering & refinement
        // Should probably be done on the connector side, but we need to stay
        // backward compatible so we still need to pass [-Infinity, Infinity]
        var values = [
            minFinite > maxRange ? maxRange : minFinite,
            maxFinite < minRange ? minRange : maxFinite
        ];
        _preact.render(_preact.h(_sliderDefault.default, {
            cssClasses: cssClasses,
            refine: refine,
            min: minRange,
            max: maxRange,
            values: values,
            tooltips: tooltips,
            step: step,
            pips: pips
        }), containerNode);
    };
};
function rangeSlider(widgetParams) {
    var _ref3 = widgetParams || {}, container = _ref3.container, attribute = _ref3.attribute, min = _ref3.min, max = _ref3.max, _ref3$cssClasses = _ref3.cssClasses, userCssClasses = _ref3$cssClasses === void 0 ? {} : _ref3$cssClasses, step = _ref3.step, _ref3$pips = _ref3.pips, pips = _ref3$pips === void 0 ? true : _ref3$pips, _ref3$precision = _ref3.precision, precision = _ref3$precision === void 0 ? 0 : _ref3$precision, _ref3$tooltips = _ref3.tooltips, tooltips = _ref3$tooltips === void 0 ? true : _ref3$tooltips;
    if (!container) throw new Error(withUsage('The `container` option is required.'));
    var containerNode = _utils.getContainerNode(container);
    var cssClasses = {
        root: _classnamesDefault.default(suit(), userCssClasses.root),
        disabledRoot: _classnamesDefault.default(suit({
            modifierName: 'disabled'
        }), userCssClasses.disabledRoot)
    };
    var specializedRenderer = renderer({
        containerNode: containerNode,
        step: step,
        pips: pips,
        tooltips: tooltips,
        renderState: {},
        cssClasses: cssClasses
    });
    var makeWidget = _connectRangeDefault.default(specializedRenderer, function() {
        return _preact.render(null, containerNode);
    });
    return _objectSpread(_objectSpread({}, makeWidget({
        attribute: attribute,
        min: min,
        max: max,
        precision: precision
    })), {}, {
        $$type: 'ais.rangeSlider',
        $$widgetType: 'ais.rangeSlider'
    });
}
exports.default = rangeSlider;

},{"preact":"26zcy","classnames":"jocGM","../../components/Slider/Slider":"7BJD7","../../connectors/range/connectRange":"abXn7","../../lib/utils":"etVYs","../../lib/suit":"du81D","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"7BJD7":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/** @jsx h */ var _preact = require("preact");
var _rheostat = require("./Rheostat");
var _rheostatDefault = parcelHelpers.interopDefault(_rheostat);
var _classnames = require("classnames");
var _classnamesDefault = parcelHelpers.interopDefault(_classnames);
var _utils = require("../../lib/utils");
var _pit = require("./Pit");
var _pitDefault = parcelHelpers.interopDefault(_pit);
function _typeof(obj1) {
    if (typeof Symbol === "function" && typeof Symbol.iterator === "symbol") _typeof = function _typeof(obj) {
        return typeof obj;
    };
    else _typeof = function _typeof(obj) {
        return obj && typeof Symbol === "function" && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj;
    };
    return _typeof(obj1);
}
function _toConsumableArray(arr) {
    return _arrayWithoutHoles(arr) || _iterableToArray(arr) || _unsupportedIterableToArray(arr) || _nonIterableSpread();
}
function _nonIterableSpread() {
    throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
}
function _unsupportedIterableToArray(o, minLen) {
    if (!o) return;
    if (typeof o === "string") return _arrayLikeToArray(o, minLen);
    var n = Object.prototype.toString.call(o).slice(8, -1);
    if (n === "Object" && o.constructor) n = o.constructor.name;
    if (n === "Map" || n === "Set") return Array.from(o);
    if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen);
}
function _iterableToArray(iter) {
    if (typeof Symbol !== "undefined" && Symbol.iterator in Object(iter)) return Array.from(iter);
}
function _arrayWithoutHoles(arr) {
    if (Array.isArray(arr)) return _arrayLikeToArray(arr);
}
function _arrayLikeToArray(arr, len) {
    if (len == null || len > arr.length) len = arr.length;
    for(var i = 0, arr2 = new Array(len); i < len; i++)arr2[i] = arr[i];
    return arr2;
}
function _extends() {
    _extends = Object.assign || function(target) {
        for(var i = 1; i < arguments.length; i++){
            var source = arguments[i];
            for(var key in source)if (Object.prototype.hasOwnProperty.call(source, key)) target[key] = source[key];
        }
        return target;
    };
    return _extends.apply(this, arguments);
}
function _classCallCheck(instance, Constructor) {
    if (!(instance instanceof Constructor)) throw new TypeError("Cannot call a class as a function");
}
function _defineProperties(target, props) {
    for(var i = 0; i < props.length; i++){
        var descriptor = props[i];
        descriptor.enumerable = descriptor.enumerable || false;
        descriptor.configurable = true;
        if ("value" in descriptor) descriptor.writable = true;
        Object.defineProperty(target, descriptor.key, descriptor);
    }
}
function _createClass(Constructor, protoProps, staticProps) {
    if (protoProps) _defineProperties(Constructor.prototype, protoProps);
    if (staticProps) _defineProperties(Constructor, staticProps);
    return Constructor;
}
function _inherits(subClass, superClass) {
    if (typeof superClass !== "function" && superClass !== null) throw new TypeError("Super expression must either be null or a function");
    subClass.prototype = Object.create(superClass && superClass.prototype, {
        constructor: {
            value: subClass,
            writable: true,
            configurable: true
        }
    });
    if (superClass) _setPrototypeOf(subClass, superClass);
}
function _setPrototypeOf(o1, p1) {
    _setPrototypeOf = Object.setPrototypeOf || function _setPrototypeOf(o, p) {
        o.__proto__ = p;
        return o;
    };
    return _setPrototypeOf(o1, p1);
}
function _createSuper(Derived) {
    var hasNativeReflectConstruct = _isNativeReflectConstruct();
    return function _createSuperInternal() {
        var Super = _getPrototypeOf(Derived), result;
        if (hasNativeReflectConstruct) {
            var NewTarget = _getPrototypeOf(this).constructor;
            result = Reflect.construct(Super, arguments, NewTarget);
        } else result = Super.apply(this, arguments);
        return _possibleConstructorReturn(this, result);
    };
}
function _possibleConstructorReturn(self, call) {
    if (call && (_typeof(call) === "object" || typeof call === "function")) return call;
    return _assertThisInitialized(self);
}
function _assertThisInitialized(self) {
    if (self === void 0) throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
    return self;
}
function _isNativeReflectConstruct() {
    if (typeof Reflect === "undefined" || !Reflect.construct) return false;
    if (Reflect.construct.sham) return false;
    if (typeof Proxy === "function") return true;
    try {
        Date.prototype.toString.call(Reflect.construct(Date, [], function() {}));
        return true;
    } catch (e) {
        return false;
    }
}
function _getPrototypeOf(o2) {
    _getPrototypeOf = Object.setPrototypeOf ? Object.getPrototypeOf : function _getPrototypeOf(o) {
        return o.__proto__ || Object.getPrototypeOf(o);
    };
    return _getPrototypeOf(o2);
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
var Slider = /*#__PURE__*/ function(_Component) {
    _inherits(Slider1, _Component);
    var _super = _createSuper(Slider1);
    function Slider1() {
        var _this;
        _classCallCheck(this, Slider1);
        for(var _len = arguments.length, args = new Array(_len), _key = 0; _key < _len; _key++)args[_key] = arguments[_key];
        _this = _super.call.apply(_super, [
            this
        ].concat(args));
        _defineProperty(_assertThisInitialized(_this), "handleChange", function(_ref) {
            var values = _ref.values;
            if (!_this.isDisabled) _this.props.refine(values);
        });
        _defineProperty(_assertThisInitialized(_this), "createHandleComponent", function(tooltips) {
            return function(props) {
                // display only two decimals after comma,
                // and apply `tooltips.format()` if any
                var roundedValue = Math.round(parseFloat(props['aria-valuenow']) * 100) / 100;
                var value = tooltips && tooltips.format ? tooltips.format(roundedValue) : roundedValue;
                var className = _classnamesDefault.default(props.className, {
                    'rheostat-handle-lower': props['data-handle-key'] === 0,
                    'rheostat-handle-upper': props['data-handle-key'] === 1
                });
                return _preact.h("div", _extends({}, props, {
                    className: className
                }), tooltips && _preact.h("div", {
                    className: "rheostat-tooltip"
                }, value));
            };
        });
        return _this;
    }
    _createClass(Slider1, [
        {
            key: "isDisabled",
            get: function get() {
                return this.props.min >= this.props.max;
            }
        },
        {
            key: "computeDefaultPitPoints",
            value: function computeDefaultPitPoints(_ref2) {
                var min = _ref2.min, max = _ref2.max;
                var totalLength = max - min;
                var steps = 34;
                var stepsLength = totalLength / steps;
                var pitPoints = [
                    min
                ].concat(_toConsumableArray(_utils.range({
                    end: steps - 1
                }).map(function(step) {
                    return min + stepsLength * (step + 1);
                })), [
                    max
                ]);
                return pitPoints;
            } // creates an array of values where the slider should snap to
        },
        {
            key: "computeSnapPoints",
            value: function computeSnapPoints(_ref3) {
                var min = _ref3.min, max = _ref3.max, step = _ref3.step;
                if (!step) return undefined;
                return [].concat(_toConsumableArray(_utils.range({
                    start: min,
                    end: max,
                    step: step
                })), [
                    max
                ]);
            }
        },
        {
            key: "render",
            value: function render() {
                var _this$props = this.props, tooltips = _this$props.tooltips, step = _this$props.step, pips = _this$props.pips, values = _this$props.values, cssClasses = _this$props.cssClasses;
                var _ref4 = this.isDisabled ? {
                    min: this.props.min,
                    max: this.props.max + 0.001
                } : this.props, min = _ref4.min, max = _ref4.max;
                var snapPoints = this.computeSnapPoints({
                    min: min,
                    max: max,
                    step: step
                });
                var pitPoints = pips === false ? [] : this.computeDefaultPitPoints({
                    min: min,
                    max: max
                });
                return _preact.h("div", {
                    className: _classnamesDefault.default(cssClasses.root, _defineProperty({}, cssClasses.disabledRoot, this.isDisabled))
                }, _preact.h(_rheostatDefault.default, {
                    handle: this.createHandleComponent(tooltips),
                    onChange: this.handleChange,
                    min: min,
                    max: max,
                    pitComponent: _pitDefault.default,
                    pitPoints: pitPoints,
                    snap: true,
                    snapPoints: snapPoints,
                    values: this.isDisabled ? [
                        min,
                        max
                    ] : values,
                    disabled: this.isDisabled
                }));
            }
        }
    ]);
    return Slider1;
}(_preact.Component);
exports.default = Slider;

},{"preact":"26zcy","./Rheostat":"1ASZX","classnames":"jocGM","../../lib/utils":"etVYs","./Pit":"6Dz8C","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"1ASZX":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/**
 * This is a fork of Rheostat for Preact X.
 *
 * @see https://github.com/airbnb/rheostat
 */ /** @jsx h */ var _preact = require("preact");
function _typeof(obj1) {
    if (typeof Symbol === "function" && typeof Symbol.iterator === "symbol") _typeof = function _typeof(obj) {
        return typeof obj;
    };
    else _typeof = function _typeof(obj) {
        return obj && typeof Symbol === "function" && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj;
    };
    return _typeof(obj1);
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
function _extends() {
    _extends = Object.assign || function(target) {
        for(var i = 1; i < arguments.length; i++){
            var source = arguments[i];
            for(var key in source)if (Object.prototype.hasOwnProperty.call(source, key)) target[key] = source[key];
        }
        return target;
    };
    return _extends.apply(this, arguments);
}
function _classCallCheck(instance, Constructor) {
    if (!(instance instanceof Constructor)) throw new TypeError("Cannot call a class as a function");
}
function _defineProperties(target, props) {
    for(var i = 0; i < props.length; i++){
        var descriptor = props[i];
        descriptor.enumerable = descriptor.enumerable || false;
        descriptor.configurable = true;
        if ("value" in descriptor) descriptor.writable = true;
        Object.defineProperty(target, descriptor.key, descriptor);
    }
}
function _createClass(Constructor, protoProps, staticProps) {
    if (protoProps) _defineProperties(Constructor.prototype, protoProps);
    if (staticProps) _defineProperties(Constructor, staticProps);
    return Constructor;
}
function _inherits(subClass, superClass) {
    if (typeof superClass !== "function" && superClass !== null) throw new TypeError("Super expression must either be null or a function");
    subClass.prototype = Object.create(superClass && superClass.prototype, {
        constructor: {
            value: subClass,
            writable: true,
            configurable: true
        }
    });
    if (superClass) _setPrototypeOf(subClass, superClass);
}
function _setPrototypeOf(o1, p1) {
    _setPrototypeOf = Object.setPrototypeOf || function _setPrototypeOf(o, p) {
        o.__proto__ = p;
        return o;
    };
    return _setPrototypeOf(o1, p1);
}
function _createSuper(Derived) {
    var hasNativeReflectConstruct = _isNativeReflectConstruct();
    return function _createSuperInternal() {
        var Super = _getPrototypeOf(Derived), result;
        if (hasNativeReflectConstruct) {
            var NewTarget = _getPrototypeOf(this).constructor;
            result = Reflect.construct(Super, arguments, NewTarget);
        } else result = Super.apply(this, arguments);
        return _possibleConstructorReturn(this, result);
    };
}
function _possibleConstructorReturn(self, call) {
    if (call && (_typeof(call) === "object" || typeof call === "function")) return call;
    return _assertThisInitialized(self);
}
function _assertThisInitialized(self) {
    if (self === void 0) throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
    return self;
}
function _isNativeReflectConstruct() {
    if (typeof Reflect === "undefined" || !Reflect.construct) return false;
    if (Reflect.construct.sham) return false;
    if (typeof Proxy === "function") return true;
    try {
        Date.prototype.toString.call(Reflect.construct(Date, [], function() {}));
        return true;
    } catch (e) {
        return false;
    }
}
function _getPrototypeOf(o2) {
    _getPrototypeOf = Object.setPrototypeOf ? Object.getPrototypeOf : function _getPrototypeOf(o) {
        return o.__proto__ || Object.getPrototypeOf(o);
    };
    return _getPrototypeOf(o2);
}
var KEYS = {
    DOWN: 40,
    END: 35,
    ESC: 27,
    HOME: 36,
    LEFT: 37,
    PAGE_DOWN: 34,
    PAGE_UP: 33,
    RIGHT: 39,
    UP: 38
};
var PERCENT_EMPTY = 0;
var PERCENT_FULL = 100;
function getPosition(value, min, max) {
    return (value - min) / (max - min) * 100;
}
function getValue(pos, min, max) {
    var decimal = pos / 100;
    if (pos === 0) return min;
    else if (pos === 100) return max;
    return Math.round((max - min) * decimal + min);
}
function getClassName(props) {
    var orientation = props.orientation === 'vertical' ? 'rheostat-vertical' : 'rheostat-horizontal';
    return [
        'rheostat',
        orientation
    ].concat(props.className.split(' ')).join(' ');
}
function getHandleFor(ev) {
    return Number(ev.currentTarget.getAttribute('data-handle-key'));
}
function killEvent(ev) {
    ev.stopPropagation();
    ev.preventDefault();
}
var Button = /*#__PURE__*/ function(_Component) {
    _inherits(Button1, _Component);
    var _super = _createSuper(Button1);
    function Button1() {
        _classCallCheck(this, Button1);
        return _super.apply(this, arguments);
    }
    _createClass(Button1, [
        {
            key: "render",
            value: function render() {
                return _preact.h("button", _extends({}, this.props, {
                    type: "button"
                }));
            }
        }
    ]);
    return Button1;
}(_preact.Component);
var _ref2 = _preact.h("div", {
    className: "rheostat-background"
});
var Rheostat = /*#__PURE__*/ function(_Component2) {
    _inherits(Rheostat1, _Component2);
    var _super2 = _createSuper(Rheostat1);
    function Rheostat1(props) {
        var _this;
        _classCallCheck(this, Rheostat1);
        _this = _super2.call(this, props);
        _defineProperty(_assertThisInitialized(_this), "state", {
            className: getClassName(_this.props),
            handlePos: _this.props.values.map(function(value) {
                return getPosition(value, _this.props.min, _this.props.max);
            }),
            handleDimensions: 0,
            mousePos: null,
            sliderBox: {},
            slidingIndex: null,
            values: _this.props.values
        });
        _this.getPublicState = _this.getPublicState.bind(_assertThisInitialized(_this));
        _this.getSliderBoundingBox = _this.getSliderBoundingBox.bind(_assertThisInitialized(_this));
        _this.getProgressStyle = _this.getProgressStyle.bind(_assertThisInitialized(_this));
        _this.getMinValue = _this.getMinValue.bind(_assertThisInitialized(_this));
        _this.getMaxValue = _this.getMaxValue.bind(_assertThisInitialized(_this));
        _this.getHandleDimensions = _this.getHandleDimensions.bind(_assertThisInitialized(_this));
        _this.getClosestSnapPoint = _this.getClosestSnapPoint.bind(_assertThisInitialized(_this));
        _this.getSnapPosition = _this.getSnapPosition.bind(_assertThisInitialized(_this));
        _this.getNextPositionForKey = _this.getNextPositionForKey.bind(_assertThisInitialized(_this));
        _this.getNextState = _this.getNextState.bind(_assertThisInitialized(_this));
        _this.handleClick = _this.handleClick.bind(_assertThisInitialized(_this));
        _this.getClosestHandle = _this.getClosestHandle.bind(_assertThisInitialized(_this));
        _this.setStartSlide = _this.setStartSlide.bind(_assertThisInitialized(_this));
        _this.startMouseSlide = _this.startMouseSlide.bind(_assertThisInitialized(_this));
        _this.startTouchSlide = _this.startTouchSlide.bind(_assertThisInitialized(_this));
        _this.handleMouseSlide = _this.handleMouseSlide.bind(_assertThisInitialized(_this));
        _this.handleTouchSlide = _this.handleTouchSlide.bind(_assertThisInitialized(_this));
        _this.handleSlide = _this.handleSlide.bind(_assertThisInitialized(_this));
        _this.endSlide = _this.endSlide.bind(_assertThisInitialized(_this));
        _this.handleKeydown = _this.handleKeydown.bind(_assertThisInitialized(_this));
        _this.validatePosition = _this.validatePosition.bind(_assertThisInitialized(_this));
        _this.validateValues = _this.validateValues.bind(_assertThisInitialized(_this));
        _this.canMove = _this.canMove.bind(_assertThisInitialized(_this));
        _this.fireChangeEvent = _this.fireChangeEvent.bind(_assertThisInitialized(_this));
        _this.slideTo = _this.slideTo.bind(_assertThisInitialized(_this));
        _this.updateNewValues = _this.updateNewValues.bind(_assertThisInitialized(_this));
        return _this;
    }
    _createClass(Rheostat1, [
        {
            key: "componentWillReceiveProps",
            value: function componentWillReceiveProps(nextProps) {
                var _this$props = this.props, className = _this$props.className, disabled = _this$props.disabled, min = _this$props.min, max = _this$props.max, orientation = _this$props.orientation;
                var _this$state = this.state, values = _this$state.values, slidingIndex = _this$state.slidingIndex;
                var minMaxChanged = nextProps.min !== min || nextProps.max !== max;
                var valuesChanged = values.length !== nextProps.values.length || values.some(function(value, idx) {
                    return nextProps.values[idx] !== value;
                });
                var orientationChanged = nextProps.className !== className || nextProps.orientation !== orientation;
                var willBeDisabled = nextProps.disabled && !disabled;
                if (orientationChanged) this.setState({
                    className: getClassName(nextProps)
                });
                if (minMaxChanged || valuesChanged) this.updateNewValues(nextProps);
                if (willBeDisabled && slidingIndex !== null) this.endSlide();
            }
        },
        {
            key: "getPublicState",
            value: function getPublicState() {
                var _this$props2 = this.props, min = _this$props2.min, max = _this$props2.max;
                var values = this.state.values;
                return {
                    max: max,
                    min: min,
                    values: values
                };
            }
        },
        {
            key: "getSliderBoundingBox",
            value: function getSliderBoundingBox() {
                var node = this.rheostat.getDOMNode ? this.rheostat.getDOMNode() : this.rheostat;
                var rect = node.getBoundingClientRect();
                return {
                    height: rect.height || node.clientHeight,
                    left: rect.left,
                    top: rect.top,
                    width: rect.width || node.clientWidth
                };
            }
        },
        {
            key: "getProgressStyle",
            value: function getProgressStyle(idx) {
                var handlePos = this.state.handlePos;
                var value = handlePos[idx];
                if (idx === 0) return this.props.orientation === 'vertical' ? {
                    height: "".concat(value, "%"),
                    top: 0
                } : {
                    left: 0,
                    width: "".concat(value, "%")
                };
                var prevValue = handlePos[idx - 1];
                var diffValue = value - prevValue;
                return this.props.orientation === 'vertical' ? {
                    height: "".concat(diffValue, "%"),
                    top: "".concat(prevValue, "%")
                } : {
                    left: "".concat(prevValue, "%"),
                    width: "".concat(diffValue, "%")
                };
            }
        },
        {
            key: "getMinValue",
            value: function getMinValue(idx) {
                return this.state.values[idx - 1] ? Math.max(this.props.min, this.state.values[idx - 1]) : this.props.min;
            }
        },
        {
            key: "getMaxValue",
            value: function getMaxValue(idx) {
                return this.state.values[idx + 1] ? Math.min(this.props.max, this.state.values[idx + 1]) : this.props.max;
            }
        },
        {
            key: "getHandleDimensions",
            value: function getHandleDimensions(ev, sliderBox) {
                var handleNode = ev.currentTarget || null;
                if (!handleNode) return 0;
                return this.props.orientation === 'vertical' ? handleNode.clientHeight / sliderBox.height * PERCENT_FULL / 2 : handleNode.clientWidth / sliderBox.width * PERCENT_FULL / 2;
            }
        },
        {
            key: "getClosestSnapPoint",
            value: function getClosestSnapPoint(value) {
                if (!this.props.snapPoints.length) return value;
                return this.props.snapPoints.reduce(function(snapTo, snap) {
                    return Math.abs(snapTo - value) < Math.abs(snap - value) ? snapTo : snap;
                });
            }
        },
        {
            key: "getSnapPosition",
            value: function getSnapPosition(positionPercent) {
                if (!this.props.snap) return positionPercent;
                var _this$props3 = this.props, max = _this$props3.max, min = _this$props3.min;
                var value = getValue(positionPercent, min, max);
                var snapValue = this.getClosestSnapPoint(value);
                return getPosition(snapValue, min, max);
            }
        },
        {
            key: "getNextPositionForKey",
            value: function getNextPositionForKey(idx, keyCode) {
                var _stepMultiplier;
                var _this$state2 = this.state, handlePos = _this$state2.handlePos, values = _this$state2.values;
                var _this$props4 = this.props, max = _this$props4.max, min = _this$props4.min, snapPoints = _this$props4.snapPoints;
                var shouldSnap = this.props.snap;
                var proposedValue = values[idx];
                var proposedPercentage = handlePos[idx];
                var originalPercentage = proposedPercentage;
                var stepValue = 1;
                if (max >= 100) proposedPercentage = Math.round(proposedPercentage);
                else stepValue = 100 / (max - min);
                var currentIndex = null;
                if (shouldSnap) currentIndex = snapPoints.indexOf(this.getClosestSnapPoint(values[idx]));
                var stepMultiplier = (_stepMultiplier = {}, _defineProperty(_stepMultiplier, KEYS.LEFT, function(v) {
                    return v * -1;
                }), _defineProperty(_stepMultiplier, KEYS.RIGHT, function(v) {
                    return v;
                }), _defineProperty(_stepMultiplier, KEYS.UP, function(v) {
                    return v;
                }), _defineProperty(_stepMultiplier, KEYS.DOWN, function(v) {
                    return v * -1;
                }), _defineProperty(_stepMultiplier, KEYS.PAGE_DOWN, function(v) {
                    return v > 1 ? -v : v * -10;
                }), _defineProperty(_stepMultiplier, KEYS.PAGE_UP, function(v) {
                    return v > 1 ? v : v * 10;
                }), _stepMultiplier);
                if (Object.prototype.hasOwnProperty.call(stepMultiplier, keyCode)) {
                    proposedPercentage += stepMultiplier[keyCode](stepValue);
                    if (shouldSnap) {
                        if (proposedPercentage > originalPercentage) // move cursor right unless overflow
                        {
                            if (currentIndex < snapPoints.length - 1) proposedValue = snapPoints[currentIndex + 1];
                             // move cursor left unless there is overflow
                        } else if (currentIndex > 0) proposedValue = snapPoints[currentIndex - 1];
                    }
                } else if (keyCode === KEYS.HOME) {
                    proposedPercentage = PERCENT_EMPTY;
                    if (shouldSnap) proposedValue = snapPoints[0];
                } else if (keyCode === KEYS.END) {
                    proposedPercentage = PERCENT_FULL;
                    if (shouldSnap) proposedValue = snapPoints[snapPoints.length - 1];
                } else return null;
                return shouldSnap ? getPosition(proposedValue, min, max) : proposedPercentage;
            }
        },
        {
            key: "getNextState",
            value: function getNextState(idx, proposedPosition) {
                var handlePos = this.state.handlePos;
                var _this$props5 = this.props, max = _this$props5.max, min = _this$props5.min;
                var actualPosition = this.validatePosition(idx, proposedPosition);
                var nextHandlePos = handlePos.map(function(pos, index) {
                    return index === idx ? actualPosition : pos;
                });
                return {
                    handlePos: nextHandlePos,
                    values: nextHandlePos.map(function(pos) {
                        return getValue(pos, min, max);
                    })
                };
            }
        },
        {
            key: "getClosestHandle",
            value: function getClosestHandle(positionPercent) {
                var handlePos = this.state.handlePos;
                return handlePos.reduce(function(closestIdx, node, idx) {
                    var challenger = Math.abs(handlePos[idx] - positionPercent);
                    var current = Math.abs(handlePos[closestIdx] - positionPercent);
                    return challenger < current ? idx : closestIdx;
                }, 0);
            }
        },
        {
            key: "setStartSlide",
            value: function setStartSlide(ev, x, y) {
                var sliderBox = this.getSliderBoundingBox();
                this.setState({
                    handleDimensions: this.getHandleDimensions(ev, sliderBox),
                    mousePos: {
                        x: x,
                        y: y
                    },
                    sliderBox: sliderBox,
                    slidingIndex: getHandleFor(ev)
                });
            }
        },
        {
            key: "startMouseSlide",
            value: function startMouseSlide(ev) {
                this.setStartSlide(ev, ev.clientX, ev.clientY);
                if (typeof document.addEventListener === 'function') {
                    document.addEventListener('mousemove', this.handleMouseSlide, false);
                    document.addEventListener('mouseup', this.endSlide, false);
                } else {
                    document.attachEvent('onmousemove', this.handleMouseSlide);
                    document.attachEvent('onmouseup', this.endSlide);
                }
                killEvent(ev);
            }
        },
        {
            key: "startTouchSlide",
            value: function startTouchSlide(ev) {
                if (ev.changedTouches.length > 1) return;
                var touch = ev.changedTouches[0];
                this.setStartSlide(ev, touch.clientX, touch.clientY);
                document.addEventListener('touchmove', this.handleTouchSlide, false);
                document.addEventListener('touchend', this.endSlide, false);
                if (this.props.onSliderDragStart) this.props.onSliderDragStart();
                killEvent(ev);
            }
        },
        {
            key: "handleMouseSlide",
            value: function handleMouseSlide(ev) {
                if (this.state.slidingIndex === null) return;
                this.handleSlide(ev.clientX, ev.clientY);
                killEvent(ev);
            }
        },
        {
            key: "handleTouchSlide",
            value: function handleTouchSlide(ev) {
                if (this.state.slidingIndex === null) return;
                if (ev.changedTouches.length > 1) {
                    this.endSlide();
                    return;
                }
                var touch = ev.changedTouches[0];
                this.handleSlide(touch.clientX, touch.clientY);
                killEvent(ev);
            }
        },
        {
            key: "handleSlide",
            value: function handleSlide(x, y) {
                var _this$state3 = this.state, idx = _this$state3.slidingIndex, sliderBox = _this$state3.sliderBox;
                var positionPercent = this.props.orientation === 'vertical' ? (y - sliderBox.top) / sliderBox.height * PERCENT_FULL : (x - sliderBox.left) / sliderBox.width * PERCENT_FULL;
                this.slideTo(idx, positionPercent);
                if (this.canMove(idx, positionPercent)) {
                    // update mouse positions
                    this.setState({
                        x: x,
                        y: y
                    });
                    if (this.props.onSliderDragMove) this.props.onSliderDragMove();
                }
            }
        },
        {
            key: "endSlide",
            value: function endSlide() {
                var _this2 = this;
                var idx = this.state.slidingIndex;
                this.setState({
                    slidingIndex: null
                });
                if (typeof document.removeEventListener === 'function') {
                    document.removeEventListener('mouseup', this.endSlide, false);
                    document.removeEventListener('touchend', this.endSlide, false);
                    document.removeEventListener('touchmove', this.handleTouchSlide, false);
                    document.removeEventListener('mousemove', this.handleMouseSlide, false);
                } else {
                    document.detachEvent('onmousemove', this.handleMouseSlide);
                    document.detachEvent('onmouseup', this.endSlide);
                }
                if (this.props.onSliderDragEnd) this.props.onSliderDragEnd();
                if (this.props.snap) {
                    var positionPercent = this.getSnapPosition(this.state.handlePos[idx]);
                    this.slideTo(idx, positionPercent, function() {
                        return _this2.fireChangeEvent();
                    });
                } else this.fireChangeEvent();
            }
        },
        {
            key: "handleClick",
            value: function handleClick(ev) {
                var _this3 = this;
                if (ev.target.getAttribute('data-handle-key')) return;
                 // Calculate the position of the slider on the page so we can determine
                // the position where you click in relativity.
                var sliderBox = this.getSliderBoundingBox();
                var positionDecimal = this.props.orientation === 'vertical' ? (ev.clientY - sliderBox.top) / sliderBox.height : (ev.clientX - sliderBox.left) / sliderBox.width;
                var positionPercent = positionDecimal * PERCENT_FULL;
                var handleId = this.getClosestHandle(positionPercent);
                var validPositionPercent = this.getSnapPosition(positionPercent); // Move the handle there
                this.slideTo(handleId, validPositionPercent, function() {
                    return _this3.fireChangeEvent();
                });
                if (this.props.onClick) this.props.onClick();
            }
        },
        {
            key: "handleKeydown",
            value: function handleKeydown(ev) {
                var _this4 = this;
                var idx = getHandleFor(ev);
                if (ev.keyCode === KEYS.ESC) {
                    ev.currentTarget.blur();
                    return;
                }
                var proposedPercentage = this.getNextPositionForKey(idx, ev.keyCode);
                if (proposedPercentage === null) return;
                if (this.canMove(idx, proposedPercentage)) {
                    this.slideTo(idx, proposedPercentage, function() {
                        return _this4.fireChangeEvent();
                    });
                    if (this.props.onKeyPress) this.props.onKeyPress();
                }
                killEvent(ev);
            } // Make sure the proposed position respects the bounds and
        },
        {
            key: "validatePosition",
            value: function validatePosition(idx, proposedPosition) {
                var _this$state4 = this.state, handlePos = _this$state4.handlePos, handleDimensions = _this$state4.handleDimensions;
                return Math.max(Math.min(proposedPosition, handlePos[idx + 1] !== undefined ? handlePos[idx + 1] - handleDimensions : PERCENT_FULL // 100% is the highest value
                ), handlePos[idx - 1] !== undefined ? handlePos[idx - 1] + handleDimensions : PERCENT_EMPTY // 0% is the lowest value
                );
            }
        },
        {
            key: "validateValues",
            value: function validateValues(proposedValues, props) {
                var _ref = props || this.props, max = _ref.max, min = _ref.min;
                return proposedValues.map(function(value, idx, values) {
                    var realValue = Math.max(Math.min(value, max), min);
                    if (values.length && realValue < values[idx - 1]) return values[idx - 1];
                    return realValue;
                });
            }
        },
        {
            key: "canMove",
            value: function canMove(idx, proposedPosition) {
                var _this$state5 = this.state, handlePos = _this$state5.handlePos, handleDimensions = _this$state5.handleDimensions;
                if (proposedPosition < PERCENT_EMPTY) return false;
                if (proposedPosition > PERCENT_FULL) return false;
                var nextHandlePosition = handlePos[idx + 1] !== undefined ? handlePos[idx + 1] - handleDimensions : Infinity;
                if (proposedPosition > nextHandlePosition) return false;
                var prevHandlePosition = handlePos[idx - 1] !== undefined ? handlePos[idx - 1] + handleDimensions : -Infinity;
                if (proposedPosition < prevHandlePosition) return false;
                return true;
            }
        },
        {
            key: "fireChangeEvent",
            value: function fireChangeEvent() {
                var onChange = this.props.onChange;
                if (onChange) onChange(this.getPublicState());
            }
        },
        {
            key: "slideTo",
            value: function slideTo(idx, proposedPosition, onAfterSet) {
                var _this5 = this;
                var nextState = this.getNextState(idx, proposedPosition);
                this.setState(nextState, function() {
                    var onValuesUpdated = _this5.props.onValuesUpdated;
                    if (onValuesUpdated) onValuesUpdated(_this5.getPublicState());
                    if (onAfterSet) onAfterSet();
                });
            }
        },
        {
            key: "updateNewValues",
            value: function updateNewValues(nextProps) {
                var _this6 = this;
                var slidingIndex = this.state.slidingIndex; // Don't update while the slider is sliding
                if (slidingIndex !== null) return;
                var max = nextProps.max, min = nextProps.min, values = nextProps.values;
                var nextValues = this.validateValues(values, nextProps);
                this.setState({
                    handlePos: nextValues.map(function(value) {
                        return getPosition(value, min, max);
                    }),
                    values: nextValues
                }, function() {
                    return _this6.fireChangeEvent();
                });
            }
        },
        {
            key: "render",
            value: function render() {
                var _this7 = this;
                var _this$props6 = this.props, children = _this$props6.children, disabled = _this$props6.disabled, Handle = _this$props6.handle, max = _this$props6.max, min = _this$props6.min, orientation = _this$props6.orientation, PitComponent = _this$props6.pitComponent, pitPoints = _this$props6.pitPoints, ProgressBar = _this$props6.progressBar;
                var _this$state6 = this.state, className = _this$state6.className, handlePos = _this$state6.handlePos, values = _this$state6.values;
                return _preact.h("div", {
                    className: className,
                    ref: function ref(_ref3) {
                        _this7.rheostat = _ref3;
                    },
                    onClick: !disabled && this.handleClick,
                    style: {
                        position: 'relative'
                    }
                }, _ref2, handlePos.map(function(pos, idx) {
                    var handleStyle = orientation === 'vertical' ? {
                        top: "".concat(pos, "%"),
                        position: 'absolute'
                    } : {
                        left: "".concat(pos, "%"),
                        position: 'absolute'
                    };
                    return _preact.h(Handle, {
                        "aria-valuemax": _this7.getMaxValue(idx),
                        "aria-valuemin": _this7.getMinValue(idx),
                        "aria-valuenow": values[idx],
                        "aria-disabled": disabled,
                        "data-handle-key": idx,
                        className: "rheostat-handle",
                        key: "handle-".concat(idx),
                        onClick: _this7.killEvent,
                        onKeyDown: !disabled && _this7.handleKeydown,
                        onMouseDown: !disabled && _this7.startMouseSlide,
                        onTouchStart: !disabled && _this7.startTouchSlide,
                        role: "slider",
                        style: handleStyle,
                        tabIndex: 0
                    });
                }), handlePos.map(function(node, idx, arr) {
                    if (idx === 0 && arr.length > 1) return null;
                    return _preact.h(ProgressBar, {
                        className: "rheostat-progress",
                        key: "progress-bar-".concat(idx),
                        style: _this7.getProgressStyle(idx)
                    });
                }), PitComponent && pitPoints.map(function(n) {
                    var pos = getPosition(n, min, max);
                    var pitStyle = orientation === 'vertical' ? {
                        top: "".concat(pos, "%"),
                        position: 'absolute'
                    } : {
                        left: "".concat(pos, "%"),
                        position: 'absolute'
                    };
                    return _preact.h(PitComponent, {
                        key: "pit-".concat(n),
                        style: pitStyle
                    }, n);
                }), children);
            }
        }
    ]);
    return Rheostat1;
}(_preact.Component);
_defineProperty(Rheostat, "defaultProps", {
    className: '',
    children: null,
    disabled: false,
    handle: Button,
    max: PERCENT_FULL,
    min: PERCENT_EMPTY,
    onClick: null,
    onChange: null,
    onKeyPress: null,
    onSliderDragEnd: null,
    onSliderDragMove: null,
    onSliderDragStart: null,
    onValuesUpdated: null,
    orientation: 'horizontal',
    pitComponent: null,
    pitPoints: [],
    progressBar: 'div',
    snap: false,
    snapPoints: [],
    values: [
        PERCENT_EMPTY
    ]
});
exports.default = Rheostat;

},{"preact":"26zcy","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"6Dz8C":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/** @jsx h */ var _preact = require("preact");
var _classnames = require("classnames");
var _classnamesDefault = parcelHelpers.interopDefault(_classnames);
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        if (enumerableOnly) symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        });
        keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = arguments[i] != null ? arguments[i] : {};
        if (i % 2) ownKeys(Object(source), true).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        });
        else if (Object.getOwnPropertyDescriptors) Object.defineProperties(target, Object.getOwnPropertyDescriptors(source));
        else ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
var Pit = function Pit(_ref) {
    var style = _ref.style, children = _ref.children;
    // first, end & middle
    var positionValue = Math.round(parseFloat(style.left));
    var shouldDisplayValue = [
        0,
        50,
        100
    ].includes(positionValue); // Children could be an array, unwrap the value if it's the case
    // see: https://github.com/developit/preact-compat/issues/436
    var value = Array.isArray(children) ? children[0] : children;
    var pitValue = Math.round(parseInt(value, 10) * 100) / 100;
    return _preact.h("div", {
        style: _objectSpread(_objectSpread({}, style), {}, {
            marginLeft: positionValue === 100 ? '-2px' : 0
        }),
        className: _classnamesDefault.default('rheostat-marker', 'rheostat-marker-horizontal', {
            'rheostat-marker-large': shouldDisplayValue
        })
    }, shouldDisplayValue && _preact.h("div", {
        className: 'rheostat-value'
    }, pitValue));
};
exports.default = Pit;

},{"preact":"26zcy","classnames":"jocGM","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"2sKTT":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/** @jsx h */ var _preact = require("preact");
var _classnames = require("classnames");
var _classnamesDefault = parcelHelpers.interopDefault(_classnames);
var _selector = require("../../components/Selector/Selector");
var _selectorDefault = parcelHelpers.interopDefault(_selector);
var _connectSortBy = require("../../connectors/sort-by/connectSortBy");
var _connectSortByDefault = parcelHelpers.interopDefault(_connectSortBy);
var _utils = require("../../lib/utils");
var _suit = require("../../lib/suit");
function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);
    if (Object.getOwnPropertySymbols) {
        var symbols = Object.getOwnPropertySymbols(object);
        if (enumerableOnly) symbols = symbols.filter(function(sym) {
            return Object.getOwnPropertyDescriptor(object, sym).enumerable;
        });
        keys.push.apply(keys, symbols);
    }
    return keys;
}
function _objectSpread(target) {
    for(var i = 1; i < arguments.length; i++){
        var source = arguments[i] != null ? arguments[i] : {};
        if (i % 2) ownKeys(Object(source), true).forEach(function(key) {
            _defineProperty(target, key, source[key]);
        });
        else if (Object.getOwnPropertyDescriptors) Object.defineProperties(target, Object.getOwnPropertyDescriptors(source));
        else ownKeys(Object(source)).forEach(function(key) {
            Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
    }
    return target;
}
function _defineProperty(obj, key, value) {
    if (key in obj) Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
    });
    else obj[key] = value;
    return obj;
}
var withUsage = _utils.createDocumentationMessageGenerator({
    name: 'sort-by'
});
var suit = _suit.component('SortBy');
var renderer = function renderer(_ref) {
    var containerNode = _ref.containerNode, cssClasses = _ref.cssClasses;
    return function(_ref2, isFirstRendering) {
        var currentRefinement = _ref2.currentRefinement, options = _ref2.options, refine = _ref2.refine;
        if (isFirstRendering) return;
        _preact.render(_preact.h("div", {
            className: cssClasses.root
        }, _preact.h(_selectorDefault.default, {
            cssClasses: cssClasses,
            currentValue: currentRefinement,
            options: options,
            setValue: refine
        })), containerNode);
    };
};
/**
 * Sort by selector is a widget used for letting the user choose between different
 * indices that contains the same data with a different order / ranking formula.
 */ var sortBy = function sortBy(widgetParams) {
    var _ref3 = widgetParams || {}, container = _ref3.container, items = _ref3.items, _ref3$cssClasses = _ref3.cssClasses, userCssClasses = _ref3$cssClasses === void 0 ? {} : _ref3$cssClasses, transformItems = _ref3.transformItems;
    if (!container) throw new Error(withUsage('The `container` option is required.'));
    var containerNode = _utils.getContainerNode(container);
    var cssClasses = {
        root: _classnamesDefault.default(suit(), userCssClasses.root),
        select: _classnamesDefault.default(suit({
            descendantName: 'select'
        }), userCssClasses.select),
        option: _classnamesDefault.default(suit({
            descendantName: 'option'
        }), userCssClasses.option)
    };
    var specializedRenderer = renderer({
        containerNode: containerNode,
        cssClasses: cssClasses
    });
    var makeWidget = _connectSortByDefault.default(specializedRenderer, function() {
        return _preact.render(null, containerNode);
    });
    return _objectSpread(_objectSpread({}, makeWidget({
        container: containerNode,
        items: items,
        transformItems: transformItems
    })), {}, {
        $$widgetType: 'ais.sortBy'
    });
};
exports.default = sortBy;

},{"preact":"26zcy","classnames":"jocGM","../../components/Selector/Selector":"8TBho","../../connectors/sort-by/connectSortBy":"3pFgJ","../../lib/utils":"etVYs","../../lib/suit":"du81D","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"8TBho":[function(require,module,exports) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/** @jsx h */ var _preact = require("preact");
var _classnames = require("classnames");
var _classnamesDefault = parcelHelpers.interopDefault(_classnames);
function Selector(_ref) {
    var currentValue = _ref.currentValue, options = _ref.options, cssClasses = _ref.cssClasses, setValue = _ref.setValue;
    return _preact.h("select", {
        className: _classnamesDefault.default(cssClasses.select),
        onChange: function onChange(event) {
            return setValue(event.target.value);
        },
        value: "".concat(currentValue)
    }, options.map(function(option) {
        return _preact.h("option", {
            className: _classnamesDefault.default(cssClasses.option),
            key: option.label + option.value,
            value: "".concat(option.value)
        }, option.label);
    }));
}
exports.default = Selector;

},{"preact":"26zcy","classnames":"jocGM","@parcel/transformer-js/src/esmodule-helpers.js":"gkKU3"}],"1MBAZ":[function() {},{}]},["jKwHT","bNKaB"], "bNKaB", "parcelRequire89ec")

//# sourceMappingURL=index.0641b553.js.map
