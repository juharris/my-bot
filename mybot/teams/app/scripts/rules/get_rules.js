// In JavaScript because couldn't get the TypeScript import to work.
import { browser } from 'webextension-polyfill-ts'
// import { RuleSettings } from './rules'

export async function getRules() /*: Promise<RuleSettings | undefined>*/ {
    let { rules } = await browser.storage.local.get('rules')
    console.debug("onhello (get_rules): rules:", rules)
    if (rules === undefined) {
        const results = await browser.storage.sync.get('rules')
        if (results === undefined || results.rules === undefined) {
            console.debug("onhello: no rules found.")
            return undefined
        }
        rules = results.rules
    }
    return rules
}