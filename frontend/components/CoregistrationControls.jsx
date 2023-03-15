import { Container } from '@mui/material'
import CoregistrationViewerMultipleAxisControl from './CoregistrationViewerMultipleAxisControl'
import { POSITION, ROTATION, SCALE } from '../constants'

export default function CoregistrationViewerControls ({ onChange }) {
  return (
    <Container sx={{ display: 'flex', flexDirection: 'column', width: '10em' }}>
      <CoregistrationViewerMultipleAxisControl
        max={2} min={0} defaultValue={1} step={0.05}
        onChange={(axis, value) => onChange(SCALE, axis, value)} title='Scale'
      />
      <CoregistrationViewerMultipleAxisControl
        max={180} min={-180} defaultValue={0} step={0.1}
        onChange={(axis, value) => onChange(ROTATION, axis, value)} title='Rotation'
      />
      <CoregistrationViewerMultipleAxisControl
        max={100} min={-100} defaultValue={0} step={0.1}
        onChange={(axis, value) => onChange(POSITION, axis, value)} title='Position'
      />
    </Container>
  )
}
