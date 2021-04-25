import { PaletteType } from '@material-ui/core'
import Container from '@material-ui/core/Container'
import FormControl from '@material-ui/core/FormControl'
import FormControlLabel from '@material-ui/core/FormControlLabel'
import Radio from '@material-ui/core/Radio'
import RadioGroup from '@material-ui/core/RadioGroup'
import { createStyles, Theme, WithStyles, withStyles } from '@material-ui/core/styles'
import Typography from '@material-ui/core/Typography'
import React from 'react'
import { browser } from 'webextension-polyfill-ts'
import { ErrorHandler } from '../error_handler'
import { getMessage } from '../i18n_helper'
import { setupUserSettings, ThemePreferenceType } from '../user'

const styles = (theme: Theme) => createStyles({
	title: {
		marginBottom: theme.spacing(1),
	},
	section: {
		marginBottom: theme.spacing(2),
	},
	buttonHolder: {
		paddingTop: '4px',
	},
	themeSelection: {
		marginLeft: theme.spacing(2),
	},
})

class Options extends React.Component<WithStyles<typeof styles>, {
	themePreference: ThemePreferenceType | '',
}> {
	private errorHandler = new ErrorHandler(undefined)

	constructor(props: any) {
		super(props)
		this.state = {
			themePreference: '',
		}

		this.handleChange = this.handleChange.bind(this)
		this.handleThemeChange = this.handleThemeChange.bind(this)
	}

	componentDidMount(): void {
		setupUserSettings(['themePreference']).then((userSettings) => {
			const { themePreference } = userSettings
			this.setState({
				themePreference,
			})
		})
	}

	handleChange(event: React.ChangeEvent<HTMLInputElement>): void {
		const value = event.target.type === 'checkbox' ? event.target.checked : event.target.value
		this.setState<never>({
			[event.target.name]: value,
		})
	}

	handleThemeChange(event: React.ChangeEvent<HTMLInputElement>) {
		const themePreference = event.target.value as PaletteType | 'device'
		this.setState({
			themePreference,
		})
		const keys = {
			themePreference,
		}
		browser.storage.local.set(keys).catch(errorMsg => {
			this.errorHandler.showError({ errorMsg })
		})
		browser.storage.sync.set(keys).catch(errorMsg => {
			this.errorHandler.showError({ errorMsg })
		})
	}

	render(): React.ReactNode {
		const { classes } = this.props

		return <Container>
			<Typography className={classes.title} component="h4" variant="h4">
				{getMessage('optionsPageTitle') || "⚙️ Options"}
			</Typography>

			<div className={classes.section}>
				<Typography component="h5" variant="h5">
					{getMessage('themeSectionTitle') || "Theme"}
				</Typography>
				<Typography component="p">
					{getMessage('themePreferenceDescription')}
				</Typography>
				<FormControl className={classes.themeSelection} component="fieldset">
					<RadioGroup aria-label="theme" name="theme" value={this.state.themePreference} onChange={this.handleThemeChange}>
						<FormControlLabel value="light" control={<Radio />} label="Light" />
						<FormControlLabel value="dark" control={<Radio />} label="Dark" />
						<FormControlLabel value="device" control={<Radio />} label="Device Preference" />
					</RadioGroup>
				</FormControl>
			</div>
		</Container >
	}
}

export default withStyles(styles)(Options)
