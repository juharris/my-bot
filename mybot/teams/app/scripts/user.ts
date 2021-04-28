import { PaletteType } from '@material-ui/core'
import { browser } from 'webextension-polyfill-ts'
import { RuleSettings } from './rules/rules'

export type ThemePreferenceType = PaletteType | 'device'

export interface UserSettings {
	themePreference: ThemePreferenceType
	rules: string
}

export async function setupUserSettings(requiredKeys: (keyof (UserSettings))[]): Promise<UserSettings> {
	const keys: Partial<UserSettings> = {
		themePreference: undefined,
		rules: undefined,
	}
	const results = await browser.storage.local.get(keys)
	let { themePreference, rules } = results
	if (requiredKeys.indexOf('themePreference') > - 1 && themePreference === undefined) {
		const r = await browser.storage.sync.get(['themePreference'])
		themePreference = r.themePreference
		if (themePreference !== undefined) {
			browser.storage.local.set({ themePreference })
		} else {
			themePreference = 'device'
		}
	}
	const result: any = { themePreference, rules }

	if (requiredKeys.indexOf('rules') > -1 && rules === undefined) {
		// browser.storage.local.remove('rules')
		// browser.storage.sync.remove('rules')
		const r = await browser.storage.sync.get(['rules'])
		rules = r.rules
		if (rules === undefined) {
			const settings: RuleSettings = {
				dateModified: new Date(),
				apps: [
					{
						comments: "For Microsoft Teams",
						urlPattern: '/poll$',
						// TODO Add JSON paths of where to get the message, sender name, etc.
						// TODO Add time range of when to respond.
						// TODO Add message age limit.
						rules: [
							{
								messageExactMatch: "Hi",
								response: "Hey, what's up?"
							},
							{
								messageExactMatch: "Good morning",
								response: "Good morning {{ FROM }}, what's up?"
							},
							{
								messagePattern: '^(hello|hey|hi)\\b.{0,10}$',
								regexFlags: 'i',
								response: "🤖 <em>This is an automated response:</em> Hey {{ FROM_FIRST_NAME }}, what's up?"
							},
						]
					},
				]
			}
			rules = settings
		} else {
			browser.storage.local.set({ rules })
		}
		result.rules = rules
	}

	return result
}
