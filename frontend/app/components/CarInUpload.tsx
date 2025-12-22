'use client'

import { useRef, useState } from 'react'
import { Camera, UploadCloud, CheckCircle, AlertCircle } from 'lucide-react'

interface CarInUploadProps {
  onUploaded?: (payload: {
    rfid_id: string
    license_plate: string | null
    suggested_slot: string | null
  }) => void
}

export default function CarInUpload({ onUploaded }: CarInUploadProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [rfid, setRfid] = useState('')
  const [status, setStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle')
  const [message, setMessage] = useState<string | null>(null)
  const [detectedPlate, setDetectedPlate] = useState<string | null>(null)

  const handleChooseFile = () => {
    fileInputRef.current?.click()
  }

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    setFile(file)
  }

  const handleUpload = async () => {
    if (!file) {
      setStatus('error')
      setMessage('Please select an image first')
      return
    }
    if (!rfid.trim()) {
      setStatus('error')
      setMessage('Please enter an RFID ID')
      return
    }

    try {
      setStatus('uploading')
      setMessage(null)
      setDetectedPlate(null)

      const form = new FormData()
      form.append('rfid_id', rfid.trim())
      form.append('image', file)

      const apiUrl = `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/v1/esp32/in-upload`
      const res = await fetch(apiUrl, {
        method: 'POST',
        body: form,
      })

      const text = await res.text()

      if (!res.ok) {
        throw new Error(text || `Upload failed (${res.status})`)
      }

      const data = text ? JSON.parse(text) : {}
      setStatus('success')
      setMessage('Uploaded successfully')
      setDetectedPlate(data.license_plate || null)

      if (onUploaded) {
        onUploaded({
          rfid_id: data.rfid_id ?? rfid.trim(),
          license_plate: data.license_plate ?? null,
          suggested_slot: data.suggested_slot ?? null,
        })
      }
    } catch (err) {
      setStatus('error')
      setMessage(err instanceof Error ? err.message : 'Upload failed')
    }
  }

  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden">
      <div className="bg-gray-800 text-white px-4 py-2 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Camera size={20} />
          <h3 className="font-semibold">CAR IN (Upload)</h3>
        </div>
      </div>

      <div className="flex flex-col p-6 space-y-4">
        <div className="w-full max-w-md aspect-video bg-gray-100 border border-dashed border-gray-300 rounded-lg flex items-center justify-center overflow-hidden self-center">
          {previewUrl ? (
            <img
              src={previewUrl}
              alt="Uploaded car in"
              className="w-full h-full object-contain"
            />
          ) : (
            <div className="text-center text-gray-400">
              <Camera size={48} className="mx-auto mb-2 opacity-60" />
              <p className="text-sm">No image uploaded</p>
              <p className="text-xs mt-1">Click the button below to upload a car image</p>
            </div>
          )}
        </div>

        <div className="flex flex-col space-y-2 max-w-md self-center w-full">
          <label className="text-sm text-gray-600">RFID ID</label>
          <input
            type="text"
            value={rfid}
            onChange={(e) => setRfid(e.target.value)}
            className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-black"
            placeholder="Enter RFID ID"
          />
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={handleFileChange}
        />

        <div className="flex flex-col sm:flex-row gap-3 self-center">
          <button
            type="button"
            onClick={handleChooseFile}
            className="inline-flex items-center space-x-2 px-4 py-2 bg-gray-200 text-gray-800 rounded-lg hover:bg-gray-300 transition-colors"
          >
            <UploadCloud size={20} />
            <span>Select Image</span>
          </button>

          <button
            type="button"
            disabled={status === 'uploading'}
            onClick={handleUpload}
            className="inline-flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            <Camera size={20} />
            <span>{status === 'uploading' ? 'Uploading...' : 'Upload to Backend'}</span>
          </button>
        </div>

        {message && (
          <div
            className={`flex items-center space-x-2 text-sm px-3 py-2 rounded ${
              status === 'success'
                ? 'bg-green-50 text-green-700 border border-green-200'
                : 'bg-red-50 text-red-700 border border-red-200'
            }`}
          >
            {status === 'success' ? <CheckCircle size={16} /> : <AlertCircle size={16} />}
            <span>{message}</span>
          </div>
        )}

        {detectedPlate && (
          <div className="text-sm text-gray-700 px-3 py-2 bg-white border rounded self-center">
            <span className="font-semibold">Detected plate:</span> {detectedPlate}
          </div>
        )}
      </div>
    </div>
  )
}
