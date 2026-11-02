const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const html = fs.readFileSync(process.argv[2], 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements = new Map();
function element() {
  return {
    value: '', textContent: '', options: [], listeners: {},
    addEventListener(name, handler) { this.listeners[name] = handler; },
    replaceChildren(...options) { this.options = options; this.value = options[0]?.value || ''; },
    add(option) { this.options.push(option); }
  };
}
for (const id of ['form', 'status', 'results', 'submit', 'expected', 'class-search', 'class-search-status']) {
  elements.set(id, element());
}
const expected = elements.get('expected');
expected.options = [
  {value: '', textContent: 'No expected class'},
  {value: '258', textContent: 'Samoyed, Samoyede'},
  {value: '248', textContent: 'Eskimo dog, husky'},
  {value: '1', textContent: 'goldfish'}
];
vm.runInNewContext(script, {
  document: {getElementById: id => elements.get(id)},
  Option: function(text, value) { this.textContent = text; this.value = value; }
});
const search = elements.get('class-search');
function filter(query) { search.value = query; search.listeners.input(); }
filter('  SAMOYED ');
assert.deepEqual(expected.options.map(o => o.value), ['', '258']);
expected.value = '258';
filter('sam');
assert.equal(expected.value, '258');
filter('dog');
assert.deepEqual(expected.options.map(o => o.value), ['', '248']);
assert.equal(expected.value, '');
assert.match(elements.get('class-search-status').textContent, /selection cleared/);
filter('not-a-class');
assert.equal(expected.options.length, 1);
assert.match(elements.get('class-search-status').textContent, /No matching/);
filter('');
assert.equal(expected.options.length, 4);
console.log('PASS: class search, case/whitespace handling, retained/cleared selection, no matches and reset');
