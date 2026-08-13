### Camera, a camera application for Linux desktops and phones.
###
### Notes for translators:
### Screen space is tight in this app. Most strings sit in overlays drawn on top
### of the live camera preview, or in fixed-width rows. Where a length limit
### matters it is stated in the comment above the string.
### Product names (COSMIC, PipeWire, GStreamer, libcamera, V4L2, DNG, JPEG, PNG,
### HDR+, Opus, AAC, MJPEG, NV12) are not translated.

# The word "Camera", used for every place the app names itself or a camera
# device: the About page header, the name the launcher and the software stores
# list the application under, the camera settings page, and the camera shortcut
# category. It is also referenced inside longer strings as { camera }, so keep it
# a single noun that reads correctly at the start of a sentence.
camera = Camera
# Row that opens the About page, and the label of the F1 shortcut. Reused in
# both places, so it has to work as a navigation row and as an action name.
about = About
# Link label in the About page pointing at the project source repository.
repository = Repository

## Capture modes, shown in the mode carousel at the bottom of the screen.

# Video recording mode. Carousel labels must be very short: the pill is sized
# per character and only two labels are visible at a time.
mode-video = Video
# Still photo mode. Same carousel length constraint.
mode-photo = Photo
# Interval photo mode that assembles the frames into a video. Same constraint.
mode-timelapse = Timelapse
# Status badge shown while the recorded timelapse video is being finalised.
# Replaces the shot counter in a small pill, so keep it short.
timelapse-saving = Saving video...
# Mode that streams the camera to other apps instead of capturing. Only listed
# when the virtual camera is enabled. Same carousel length constraint.
mode-virtual = Virtual
# Preview only mode with no capture controls. Same carousel length constraint.
mode-view = View

## Virtual camera, a device other applications can read this camera from.

# Toggle that enables the virtual camera. Also used as the settings row and as
# the page title, so it must read well as a heading too.
virtual-camera-title = Virtual camera (experimental)
# Description under the virtual camera toggle.
virtual-camera-description = Stream your camera feed to other applications via a virtual camera device. Requires PipeWire.
# Badge shown next to a green dot while the virtual camera is streaming.
# Very short, it sits in a small pill in the top bar. Uppercase in English.
streaming-live = LIVE
# Name of the file type filter in the file chooser used to pick media to stream
# through the virtual camera.
virtual-camera-file-filter-name = Images and Videos

## Filters, GPU effects applied to the preview.

# Title of the filter picker panel.
filters-title = Filters
# Filter names, shown under thumbnails in a grid several columns wide. Each is
# one short word, so prefer a short term over an exact one.
# No filter, the unaltered image.
filter-standard = Original
# Black and white.
filter-mono = Mono
# Warm brown tint, like an aged photograph.
filter-sepia = Sepia
# High contrast black and white, in the film noir style.
filter-noir = Noir
# Boosted colour and contrast.
filter-vivid = Vivid
# Shifted towards blue.
filter-cool = Cool
# Shifted towards orange.
filter-warm = Warm
# Washed out, reduced contrast.
filter-fade = Fade
# Maps the image onto two colours.
filter-duotone = Duotone
# Darkens the corners of the image.
filter-vignette = Vignette
# Inverts the colours, like a photographic negative.
filter-negative = Negative
# Reduces the image to a few flat colour bands.
filter-posterize = Posterize
# Partially inverts the brightest areas.
filter-solarize = Solarize
# Splits the colour channels apart at the edges, imitating a lens fault. Short
# for chromatic aberration.
filter-chroma = Chroma
# Imitates a pencil sketch.
filter-pencil = Pencil

## Settings.

# Title of the settings panel.
settings-title = Settings
# Section title for the theme options at the top of the settings panel.
settings-appearance = Appearance
# Section title for the entries that are not camera settings.
settings-other = Other
# Settings row and page title for the overlay effect and composition guide.
settings-overlay = Overlay
# Dropdown label for the light or dark theme choice.
settings-theme = Theme
settings-controls-position = Controls position
settings-controls-position-description = Place the capture controls at the bottom, left, or right edge. Left and right also move the title bar to the opposite edge without rotating the camera preview.
controls-position-bottom = Bottom
controls-position-left = Left
controls-position-right = Right
# Theme option: follow the desktop light or dark preference.
match-desktop = Match Desktop
# Theme option: always dark.
dark = Dark
# Theme option: always light.
light = Light
# Dropdown label for how the on screen controls are drawn over the preview.
settings-overlay-effect = Overlay effect
# Description under the overlay effect dropdown.
settings-overlay-effect-description = How controls are painted over the preview. Frosted glass blurs the preview behind them, which uses more GPU.
# Overlay option: follow the COSMIC desktop setting. Only offered on COSMIC.
overlay-effect-system = System
# Overlay option: blur whatever is behind the controls.
overlay-effect-frosted = Frosted Glass
# Overlay option: semi transparent controls with no blur.
overlay-effect-translucent = Translucent
# Overlay option: fully opaque controls.
overlay-effect-off = Off
# Dropdown label for which capture mode the app starts in.
settings-default-mode = Default mode
# Description under the default mode dropdown.
settings-default-mode-description = Camera mode to use when the app launches
# Settings row, page title and section title for video recording options.
settings-video = Video
# Label of the camera selection row. This row also holds an info button and the
# device dropdown, so the label must stay short.
settings-device = Device
# Dropdown label for the audio input device used when recording video.
settings-microphone = Microphone
# Toggle that records audio alongside video.
settings-record-audio = Record audio
# Dropdown label for the audio codec used in recordings.
settings-audio-encoder = Audio encoder
# Label of the live microphone meter row. The row also holds the meter and a
# decibel readout, so the label must stay short.
settings-mic-level = Microphone level
# Shown in place of the microphone meter until the first audio level arrives.
settings-mic-level-initializing = Initializing…
# Dropdown label for the video codec used in recordings.
settings-encoder = Encoder
# Dropdown label for the recording bitrate preset.
settings-quality = Quality
# Toggle that flips the preview horizontally, like a mirror.
settings-mirror-preview = Mirror preview
# Description under the mirror preview toggle.
settings-mirror-preview-description = Flip the camera preview horizontally
# Toggle that also applies the mirroring to saved files. Only shown when mirror
# preview is on.
settings-mirror-captures = Mirror captures
# Description under the mirror captures toggle.
settings-mirror-captures-description = Apply the same horizontal flip to saved photos, videos, and timelapse output
# Toggle for vibration feedback. Only shown on devices that support it.
settings-haptic-feedback = Haptic feedback
# Description under the haptic feedback toggle.
settings-haptic-feedback-description = Vibrate on capture, mode switch, and camera switch
# Standalone button that restores every setting to its default, and the label
# of the matching shortcut. Reused in both places.
settings-reset-all = Reset all settings
# Settings row, page title and section title for the bug reporting tools.
settings-bug-reports = Bug reports
# Button that generates a diagnostic report.
settings-report-bug = Report bug
# Button that opens the most recently generated report. Sits beside the button
# above, so keep both short.
settings-show-report = Show Report

## Device information panel, expanded from the camera row in settings.
## These are labels in a two column list. Values are technical and untranslated.

# Label for the V4L2 card name of the camera.
device-info-card = Card
# Label for the kernel driver name.
device-info-driver = Driver
# Label for the device node path, for example /dev/video0.
device-info-path = Path
# Label for the resolved path, shown only when it differs from the path above.
device-info-real-path = Real Path
# Label for the device path reported by libcamera.
device-info-device-path = Device Path
# Label for the image sensor model, for example IMX376.
device-info-sensor = Sensor
# Label for the libcamera pipeline handler in use.
device-info-pipeline = Pipeline
# Label for the libcamera version string.
device-info-libcamera-version = libcamera
# Label for whether the camera can run preview and capture streams at once.
device-info-multistream = Multi-stream
# Value for the row above when multiple streams are possible.
device-info-multistream-yes = Supported
# Value for the row above when they are not.
device-info-multistream-no = Not supported
# Label for the sensor mounting rotation, shown in degrees.
device-info-rotation = Rotation
# Shown instead of the panel when nothing could be read from the device.
device-info-none = No device information available

## Camera preview.

# Centred placeholder shown before any camera has been found.
initializing-camera = Initializing camera...

## Format picker, the resolution and frame rate chooser.

# Label before the row of resolution buttons. Includes the colon. Fixed 80px
# label column, so keep it short.
format-resolution = Resolution:
# Label before the row of frame rate buttons. Includes the colon. Same 80px
# label column.
format-framerate = Frame Rate:

## Status indicators in the format button in the top bar.
## These are tiny badges, 2 to 4 characters. Abbreviate.

# Superscript after the resolution class, short for resolution.
indicator-res = RES
# Superscript after the frame rate number, short for frames per second.
indicator-fps = FPS
# Resolution badge for high definition, roughly 1080p.
indicator-hd = HD
# Resolution badge for anything below 720p.
indicator-sd = SD
# Resolution badge for 4K. Usually left as is.
indicator-4k = 4K
# Resolution badge for 720p. Usually left as is.
indicator-720p = 720p

## Actions offered when a QR code is detected in the preview.
## Each is a button in a small overlay near the code. Short verb phrases.

# Opens a detected web address.
qr-open-link = Open Link
# Joins the wireless network described by the code.
qr-connect-wifi = Connect to WiFi
# Copies plain text content to the clipboard.
qr-copy-text = Copy Text
# Dials a detected phone number.
qr-call = Call
# Starts a new message to a detected email address.
qr-send-email = Send Email
# Starts a new text message to a detected number.
qr-send-sms = Send SMS
# Opens detected coordinates in a map application.
qr-open-map = Open Map
# Saves a detected contact card.
qr-add-contact = Add Contact
# Saves a detected calendar entry.
qr-add-event = Add Event

## Exposure picker.
## Row labels here sit in a fixed 70px column at text size 13. Long words are
## clipped, so abbreviate where needed.

# Slider label for exposure compensation, in EV stops. EV is a photographic
# abbreviation and is usually kept.
exposure-ev = EV
# Slider label for the shutter speed, shown as a fraction of a second.
exposure-time = Time
# Slider label for sensor gain.
exposure-gain = Gain
# Slider label for ISO sensitivity. ISO is usually kept as is.
exposure-iso = ISO
# Label before the light metering mode buttons.
exposure-metering = Metering
# Toggle that lets the camera lower the frame rate to allow longer exposures in
# low light. It does not display the frame rate.
exposure-auto-priority = Variable Frame Rate
# Slider label for backlight compensation. Shown in automatic mode only.
exposure-backlight = Backlight
# Segmented button option for manual exposure. Shares its width with the option
# below, so both must be very short.
exposure-manual-mode = Manual
# Segmented button option for automatic exposure. Same width constraint.
exposure-auto-mode = Auto
# Shown beside a control the connected camera does not offer. Lowercase in
# English because it reads as a status, not a heading.
exposure-not-supported = unsupported

## Focus controls, part of the exposure picker. Same 70px label column.

# Toggle for automatic focus.
focus-auto = Autofocus
# Slider label for the manual focus distance. Shown when autofocus is off.
focus-position = Focus Position

## Colour picker. Same 70px label column.

# Title of the colour picker panel.
color-title = Color
# Slider label for image contrast.
color-contrast = Contrast
# Slider label for colour intensity.
color-saturation = Saturation
# Slider label for edge sharpening.
color-sharpness = Sharpness
# Slider label for colour hue.
color-hue = Hue
# Toggle for automatic white balance.
color-white-balance = White Balance
# Slider label for the manual white balance temperature in Kelvin. Shown when
# automatic white balance is off. Abbreviated to fit the 70px column.
color-temperature = Temp
# Status text beside the autofocus and white balance toggles when they are on.
color-auto = Auto
# Status text beside those toggles when they are off.
color-manual = Manual

## Tools grid, the row of buttons above the shutter.
## These are labels under 32px icons at text size 11. One short word each.

# Opens the self timer. Photo mode only.
tools-timer = Timer
# Cycles the photo aspect ratio. Photo mode only.
tools-aspect = Aspect
# Opens the exposure picker.
tools-exposure = Exposure
# Opens the colour picker.
tools-color = Color
# Opens the filter picker.
tools-filter = Filter
# Opens the pan and tilt controls. Only shown when the camera has a motor.
tools-motor = Motor

## Pan and tilt controls for motorised cameras.

# Title of the pan and tilt panel.
ptz-title = Camera Controls
# Slider label for horizontal camera rotation, shown in degrees. Sits in a
# fixed 60px column, so keep it short.
ptz-pan = Pan
# Slider label for vertical camera rotation, shown in degrees. Same 60px
# column.
ptz-tilt = Tilt

## Privacy cover warning, a modal shown when the lens is physically covered.

# Title of the warning. Large bold text, keep it to one line.
privacy-cover-closed = Privacy cover is closed
# Body of the warning, referring to the physical shutter on the device.
privacy-cover-hint = Open the privacy cover to use the camera

## HDR+ burst capture, which merges several frames into one photo.

# Full screen status while the frames are being taken. This is the largest text
# in the app, so keep it very short.
burst-mode-hold-steady = Hold steady...
# Progress under the status above. $captured is how many frames are done and
# $total is how many are planned.
burst-mode-frames = { $captured }/{ $total } frames
# Full screen status while the frames are merged. Same large text, keep short.
burst-mode-processing = Processing...
# Merge algorithm option: slower, better results. FFT is a technical term.
burst-mode-quality = Quality (FFT)
# Merge algorithm option: faster, lower quality.
burst-mode-fast = Fast (Spatial)

## HDR+ frame count options in settings.

# HDR+ disabled.
hdr-plus-off = Off
# Frame count chosen automatically from how bright the scene is.
hdr-plus-auto = Auto
# Fixed count of 4 frames.
hdr-plus-frames-4 = 4 frames
# Fixed count of 6 frames.
hdr-plus-frames-6 = 6 frames
# Fixed count of 8 frames.
hdr-plus-frames-8 = 8 frames
# Fixed count of 50 frames.
hdr-plus-frames-50 = 50 frames

## Photo settings.

# Settings row, page title and section title for photo options.
settings-photo = Photo
# Dropdown label for the file format photos are saved in.
settings-photo-format = Output format
# Description under the output format dropdown. The format names are not
# translated.
settings-photo-format-description = File format for saved photos. JPEG is compressed, PNG is lossless, DNG preserves raw data for editing.
# Dropdown label for the HDR+ frame count. HDR+ is a product name, keep it.
settings-hdr-plus = HDR+ (experimental)
# Description under the HDR+ dropdown.
settings-hdr-plus-description = Multi-frame capture for improved low-light photos and dynamic range. Auto selects frame count based on scene brightness.
# Toggle that also keeps every individual burst frame. Only shown when HDR+ is
# enabled.
settings-save-burst-raw = Save raw burst frames
# Description under the toggle above.
settings-save-burst-raw-description = Save individual burst frames as DNG files alongside HDR+ photos. Useful for debugging or reprocessing.

## Composition guides, optional lines drawn over the preview to help framing.

# Dropdown label for the guide overlay.
settings-composition-guide = Composition guide
# Description under the composition guide dropdown.
settings-composition-guide-description = Overlay guide lines on the camera preview for framing
# Guide option: no lines.
guide-none = None
# Guide option: a 3 by 3 grid.
guide-rule-of-thirds = Rule of Thirds
# Guide option: a grid based on the golden ratio.
guide-phi-grid = Phi Grid
# Guide option: golden spiral, curling towards the top left. Keep the arrow.
guide-spiral-top-left = Golden Ratio ↖
# Guide option: golden spiral, curling towards the top right. Keep the arrow.
guide-spiral-top-right = Golden Ratio ↗
# Guide option: golden spiral, curling towards the bottom left. Keep the arrow.
guide-spiral-bottom-left = Golden Ratio ↙
# Guide option: golden spiral, curling towards the bottom right. Keep the arrow.
guide-spiral-bottom-right = Golden Ratio ↘
# Guide option: diagonal lines.
guide-diagonal = Diagonals
# Guide option: a crosshair at the centre.
guide-crosshair = Crosshair

## About page.

# Link label pointing at the issue tracker.
about-support = Support & Feedback

## Insights, a diagnostics panel for developers and bug reports.
## Most rows here are labels beside technical values. The values themselves are
## not translated. Abbreviations and technical terms are expected.

# Title of the insights panel, and the settings row that opens it.
insights-title = Insights
# Section title for the media pipeline.
insights-pipeline = Pipeline
# Label of the row holding the full pipeline description and its copy button.
# Despite the key name this is shown for every backend, not only libcamera, so
# do not mention libcamera in the translation.
insights-pipeline-full-libcamera = Pipeline
# Label introducing the list of decoders the app falls back through.
insights-decoder-chain = Decoder Fallback Chain

# Section title used when one stream serves both preview and capture.
insights-stream-combined = Preview + Capture Stream

# Row label, delay from sensor to screen, in milliseconds.
insights-frame-latency = Frame Latency
# Row label, count of frames that never arrived.
insights-dropped-frames = Dropped Frames
# Row label, size of one decoded frame in memory, in megabytes.
insights-frame-size-decoded = Frame Size
# Row label, time spent wrapping a frame for the renderer, in milliseconds.
insights-copy-time = Frame Wrap Time
# Row label, time spent sending a frame to the GPU, in milliseconds.
insights-gpu-upload-time = GPU Upload Time
# Row label, throughput of those uploads, in megabytes per second.
insights-gpu-upload-bandwidth = GPU Upload Bandwidth

# Row label, where the video data originates.
insights-format-source = Source
# Row label, pixel dimensions of the format.
insights-format-resolution = Resolution
# Row label, frames per second of the format.
insights-format-framerate = Framerate
# Row label, the format the sensor delivers before conversion.
insights-format-native = Native Format
# Row label, processing steps performed on the CPU.
insights-cpu-processing = CPU Processing
# Row label, time the CPU spent decoding, in milliseconds.
insights-cpu-decode-time = CPU Decode Time
# Row label, processing steps performed on the GPU.
insights-format-wgpu = GPU Processing

# Status in the decoder list: this decoder is the one in use.
insights-selected = Selected
# Status in the decoder list: usable but not chosen.
insights-available = Available
# Status in the decoder list: not usable on this system.
insights-unavailable = Unavailable

## Insights, capture backend.

# Section title for the capture backend.
insights-backend = Backend
# Row label, which backend is in use, for example libcamera or V4L2.
insights-backend-type = Type
# Row label, the libcamera pipeline handler driving the sensor.
insights-pipeline-handler = Pipeline Handler
# Row label, the libcamera library version.
insights-libcamera-version = libcamera Version
# Row label, the image sensor model.
insights-sensor-model = Sensor
# Row label, which component decodes MJPEG frames.
insights-mjpeg-decoder = MJPEG Decoder

## Insights, stream layout.

# Row label used when preview and capture share one stream. Its value is the
# shared source string below.
insights-multistream-single = Single-stream
# Row label used when preview and capture have separate streams. Its value is
# the separate source string below.
insights-multistream-dual = Dual-stream
# Value beside the single stream label above.
insights-multistream-source-shared = Preview & Capture
# Value beside the dual stream label above.
insights-multistream-source-separate = Preview / Capture
# Section title for the stream feeding the on screen preview.
insights-stream-preview = Preview Stream
# Section title for the stream feeding photo and video capture.
insights-stream-capture = Capture Stream
# Row label, what the stream is used for.
insights-stream-role = Role
# Row label, pixel dimensions of the stream.
insights-stream-resolution = Resolution
# Row label, the pixel layout, for example NV12.
insights-stream-pixel-format = Pixel Format
# Row label, how many frames have passed through the stream.
insights-stream-frame-count = Frames

## Insights, video recording. Only shown while recording.

# Section title for the recording pipeline.
insights-recording = Recording Pipeline
# Row label, which recording mode is active.
insights-recording-mode = Mode
# Row label, the video codec in use.
insights-recording-encoder = Encoder
# Row label, dimensions and frame rate of the recording.
insights-recording-resolution = Resolution
# Row label, frames sent and dropped by the capture thread.
insights-recording-capture = Capture Thread
# Row label, how many frames are waiting in the queue.
insights-recording-channel = Channel
# Row label, frames pushed into GStreamer and frames skipped. Appsrc is a
# GStreamer element name and is not translated.
insights-recording-pusher = Appsrc Pusher
# Row label, the frame rate actually achieved.
insights-recording-fps = Effective FPS
# Row label, time between capturing and encoding a frame, in milliseconds.
insights-recording-delay = Processing Delay
# Row label, time spent converting to NV12, in milliseconds. NV12 is a pixel
# format name and is not translated.
insights-recording-convert = NV12 Convert
# Row label, the presentation timestamp of the newest frame, in seconds. PTS is
# a video term and is usually kept.
insights-recording-pts = Current PTS
# Row label introducing the full recording pipeline text below it.
insights-recording-pipeline = Pipeline

## Insights, audio.

# Section title for audio.
insights-audio = Audio
# Row label, whether audio is being recorded. Its value is one of the two
# strings below.
insights-audio-recording = Recording
# Row label, the selected input device.
insights-audio-device = Device
# Row label, the PipeWire or PulseAudio node name.
insights-audio-node = Node
# Row label, the audio codec, for example Opus or AAC.
insights-audio-codec = Codec
# Row label, how many channels the output has.
insights-audio-channels = Channels
# Value for the recording row: audio will be recorded.
insights-audio-enabled = Enabled
# Value for the recording row: audio will not be recorded. Also shown in place
# of the pipeline description when audio is off.
insights-audio-disabled = Disabled
# Suffix appended after a device name to mark the system default, for example
# "Built-in Audio (Default)". It is not a row of its own, so keep the brackets.
insights-audio-default = (Default)
# Single channel audio. Used both as the value of the channels row and as the
# label of the output meter, where it sits in a 48px column.
insights-audio-mono = Mono
# Row label introducing the audio pipeline description below it.
insights-audio-pipeline = Pipeline
# Row label, the sample format, rate and channel count of the device.
insights-audio-format = Format
# Row label introducing the per channel input meters below it.
insights-audio-inputs = Input Channels
# Row label introducing the mixed output meter below it.
insights-audio-output-level = Output Level

## Insights, per frame sensor metadata reported by libcamera.

# Section title for the metadata list.
insights-metadata = Frame Metadata
# Row label, the shutter time of the frame, shown in microseconds,
# milliseconds or seconds depending on length.
insights-meta-exposure = Exposure
# Row label, sensor gain applied before digitising, shown as a multiplier.
insights-meta-analogue-gain = Analogue Gain
# Row label, gain applied after digitising, shown as a multiplier.
insights-meta-digital-gain = Digital Gain
# Row label, the estimated colour temperature in Kelvin.
insights-meta-colour-temp = Colour Temp
# Row label, the frame counter from the sensor.
insights-meta-sequence = Sequence
# Row label, the red and blue white balance multipliers. WB is short for white
# balance, and R and B for red and blue.
insights-meta-colour-gains = WB Gains (R, B)
# Row label, the sensor black point.
insights-meta-black-level = Black Level
# Row label, the focus motor position, shown in dioptres.
insights-meta-lens-position = Lens Position
# Row label, the measured scene brightness in lux.
insights-meta-lux = Illuminance
# Row label, the autofocus sharpness score. FoM is short for figure of merit.
insights-meta-focus-fom = Focus FoM
# Shown as the value of any metadata row the sensor did not report. Short for
# not available.
insights-meta-na = N/A

## Timelapse settings.

# Settings row, page title and section title for timelapse options.
settings-timelapse = Timelapse
# Dropdown label for the gap between shots, in seconds.
settings-timelapse-interval = Interval
# Description under the interval dropdown.
settings-timelapse-interval-description = Time between consecutive photo captures

## Insights, V4L2 format list. Each row is one resolution the kernel driver
## reports, marked with whether libcamera also offers it.

# Status for a resolution libcamera offers. Shown after a check mark.
insights-v4l2-in-libcamera = Available in libcamera
# Status for the resolution currently in use. Shown after a check mark and
# highlighted.
insights-v4l2-active-in-libcamera = Active in libcamera
# Status for a resolution libcamera does not offer. Shown after a cross and in
# the error colour.
insights-v4l2-not-in-libcamera = Not available in libcamera

## Insights, manual capture buttons for debugging.

# Button that saves a single frame.
insights-capture = Capture
# Button that saves a burst of frames.
insights-capture-burst = Capture Burst

## Keyboard shortcut categories, the section headings on the shortcuts page.

# Groups the shutter and snapshot shortcuts.
shortcut-category-capture = Capture
# Groups the shortcuts that open the exposure, colour, motor, format and
# settings panels.
shortcut-category-pickers = Pickers
# Groups the mode switching shortcuts.
shortcut-category-display = Display
# Groups the zoom and aspect ratio shortcuts.
shortcut-category-zoom = Zoom & framing
# Groups gallery, about, reset and quit shortcuts.
shortcut-category-app = App

## Keyboard shortcut action names, listed beside their key combinations.
## Each names an action rather than commanding it.

# Takes a photo, starts or stops recording, or pauses playback depending on the
# current mode. The slashes separate the three meanings.
action-capture = Capture / Record / Play-Pause
# Takes a still photo without interrupting an ongoing video recording. Only
# works while recording.
action-photo-snapshot = Photo during recording
# Switches between the front and back camera.
action-switch-camera = Switch camera
# Turns autofocus on or off.
action-toggle-focus-auto = Toggle auto focus
# Turns the flash on or off.
action-toggle-flash = Toggle flash
# Shows or hides the tools menu holding timer, aspect, exposure, colour and
# filter shortcuts.
action-toggle-tools-menu = Toggle tools menu
# Shows or hides the exposure panel.
action-toggle-exposure-picker = Toggle exposure picker
# Shows or hides the colour panel.
action-toggle-color-picker = Toggle color picker
# Shows or hides the pan and tilt panel.
action-toggle-motor-picker = Toggle motor controls
# Shows or hides the resolution and frame rate panel.
action-toggle-format-picker = Toggle format picker
# Shows or hides the filter picker.
action-toggle-filters = Toggle filter picker
# Opens the settings panel.
action-toggle-settings = Open settings
# Moves to the next capture mode in the carousel.
action-next-mode = Next mode
# Moves to the previous capture mode in the carousel.
action-prev-mode = Previous mode
# Hides every on-screen control so only the live preview remains, or brings
# them all back.
action-toggle-ui-chrome = Toggle interface
# Zooms the preview in.
action-zoom-in = Zoom in
# Zooms the preview out.
action-zoom-out = Zoom out
# Returns the zoom to its default level.
action-reset-zoom = Reset zoom
# Switches the preview between filling the window and showing the whole frame.
action-toggle-preview-fit = Toggle fit / fill
# Steps through the available photo aspect ratios.
action-cycle-photo-aspect-ratio = Cycle photo aspect ratio
# Opens saved photos and videos in the system gallery.
action-open-gallery = Open gallery
# Opens this keyboard shortcuts page.
action-show-shortcuts = Show shortcuts
# Closes the application.
action-quit-app = Quit

## Keyboard shortcuts page and the dialog for recording a new shortcut.

# Placeholder on the button of an action that has no shortcut assigned. This is
# an em dash and normally needs no translation.
shortcuts-help-unbound = —
# Title of the shortcuts page, and the settings row that opens it.
keybindings-page-title = Keyboard Shortcuts
# Button that restores every shortcut to its default.
keybindings-page-reset-all = Reset all to defaults
# Heading of the dialog that waits for a key combination to be pressed.
keybindings-record-title = Press a key combination
# Instructions under the heading above. Esc is a key name, keep it as is.
keybindings-record-hint = Press the key combination you want, or Esc to cancel
# Warning shown when the pressed combination already belongs to another action.
# $other is that action's name, taken from the list above.
keybindings-record-conflict = This combination is already used by "{$other}". Replace?
# Confirms overwriting the conflicting shortcut. Replaces the save button when
# there is a conflict.
keybindings-record-replace = Replace
# Abandons the dialog without changing the shortcut.
keybindings-record-cancel = Cancel
# Confirms the new shortcut. Shown when there is no conflict.
keybindings-record-save = Save

## Desktop launcher entry. These strings, and the software centre ones below,
## are generated into resources/*.desktop and resources/*.metainfo.xml by
## scripts/gen-metadata.py, so those two files are never edited by hand. They
## are the only strings here that are not shown inside the application itself.

# One line tooltip shown in the launcher.
desktop-comment = Take photos, record video, and scan QR codes
# Search terms for the launcher, separated by semicolons and ending with one.
# Translate the terms, and feel free to add extra words people in your language
# would search for. Keeping the English words as well is useful, since many
# users type them regardless.
desktop-keywords = camera;webcam;photo;video;

## Software centre listing, shown by GNOME Software, KDE Discover and Flathub.
## The release notes in the metainfo file are deliberately not translated.

# One line summary shown under the name in a software centre. Keep it under
# about 60 characters, as listings truncate it. Software centres have no
# equivalent of the launcher's generic name, so this line carries the word
# "camera" for people searching the catalogue for one.
metainfo-summary = Camera for photos, videos and QR codes
# First paragraph of the long description.
metainfo-description-intro = { camera } is a modern camera application for Linux, built for desktops and phones alike. Whether you need to snap a quick photo, record a video, or scan a QR code, { camera } provides a clean and intuitive interface that stays out of your way.
# Second paragraph of the long description.
metainfo-description-usage = Just open the app and start capturing moments. Add fun filters to your photos, scan QR codes to open links or connect to WiFi, or use virtual camera mode to look great in video calls with your favorite filter applied.
# Heading introducing the feature list below. Ends with a colon.
metainfo-description-features-title = Key features:
# Feature list item.
metainfo-feature-capture = Photo and video capture: high quality and hardware accelerated
# Feature list item. Names the three capture modes, which are also translated
# in the mode carousel above, so keep the wording the same.
metainfo-feature-modes = Photo, video and timelapse modes, with a self timer and composition guides
# Feature list item.
metainfo-feature-controls = Manual controls: exposure, ISO, shutter, focus and white balance
# Feature list item. QR and WiFi are product names and stay as they are.
metainfo-feature-qr = QR code scanner: open links, connect to WiFi, and more
# Feature list item. The filter names are pulled straight from the filter
# picker above, so translate them there and they follow here on their own.
metainfo-feature-filters = 14 creative filters: { filter-mono }, { filter-sepia }, { filter-vivid }, { filter-noir }, { filter-pencil }, and more
# Feature list item.
metainfo-feature-virtual-camera = Virtual camera: use your filtered camera feed in video calls and other apps
# Feature list item.
metainfo-feature-multi-camera = Multi-camera support: easily switch between cameras and microphones

## Screenshot captions in the software centre listing.

# Caption for the screenshot of photo mode with the tools row open.
metainfo-caption-photo-tools = Photo mode with tools menu
# Caption for the screenshot taken on a mobile device.
metainfo-caption-phone = Photo mode on a Linux phone
# Caption for the screenshot of the filter picker.
metainfo-caption-filters = Filter picker
# Caption for the screenshot taken while a video is being recorded.
metainfo-caption-recording = Video recording in progress
# Caption for the screenshot showing a detected QR code.
metainfo-caption-qr = QR code detection
# Caption for the screenshot of the settings panel.
metainfo-caption-settings = Advanced settings
