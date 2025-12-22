'use client'

import { CreditCard, Car, Clock } from 'lucide-react'

export interface CarEvent {
  id: string
  rfid_id: string
  license_plate: string | null
  event_type: string
  created_at: string
  parking_slot: string | null
}

interface CarInfoProps {
  carInEvent: CarEvent | null
  carOutEvent: CarEvent | null
  loading?: boolean
}

export default function CarInfo({ carInEvent, carOutEvent, loading }: CarInfoProps) {
  if (loading) {
    return (
      <div className="bg-white p-6 rounded-lg shadow-md">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <p className="mt-2 text-gray-600">Loading car information...</p>
        </div>
      </div>
    )
  }

  const latestEvent = carInEvent || carOutEvent

  if (!latestEvent) {
    return (
      <div className="bg-white p-6 rounded-lg shadow-md">
        <div className="text-center text-gray-500">
          <Car size={48} className="mx-auto mb-4 opacity-50" />
          <p>No recent car activity</p>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white p-6 rounded-lg shadow-md">
      <h3 className="text-xl font-bold mb-4 flex items-center space-x-2">
        <Car size={24} />
        <span>Recent Car Information</span>
      </h3>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Car IN Information */}
        <div className="border-l-4 border-green-500 pl-4">
          <div className="flex items-center space-x-2 mb-3">
            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
            <h4 className="font-semibold text-green-700">CAR IN</h4>
          </div>
          
          {carInEvent ? (
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                <CreditCard size={16} className="text-gray-500" />
                <span className="text-sm text-gray-600">RFID ID:</span>
                <span className="font-semibold">{carInEvent.rfid_id}</span>
              </div>
              
              {carInEvent.license_plate && carInEvent.license_plate !== 'N/A' && (
                <div className="flex items-center space-x-2">
                  <Car size={16} className="text-gray-500" />
                  <span className="text-sm text-gray-600">License Plate:</span>
                  <span className="font-semibold">{carInEvent.license_plate}</span>
                </div>
              )}
              
              {carInEvent.parking_slot && carInEvent.parking_slot !== 'N/A' && (
                <div className="flex items-center space-x-2">
                  <span className="text-sm text-gray-600">Parking Slot:</span>
                  <span className="font-semibold bg-blue-100 text-blue-700 px-2 py-1 rounded">
                    {carInEvent.parking_slot}
                  </span>
                </div>
              )}
              
              <div className="flex items-center space-x-2">
                <Clock size={16} className="text-gray-500" />
                <span className="text-sm text-gray-600">
                  {new Date(carInEvent.created_at).toLocaleString()}
                </span>
              </div>
            </div>
          ) : (
            <p className="text-gray-400 text-sm">No car entry detected</p>
          )}
        </div>

        {/* Car OUT Information */}
        <div className="border-l-4 border-red-500 pl-4">
          <div className="flex items-center space-x-2 mb-3">
            <div className="w-3 h-3 bg-red-500 rounded-full"></div>
            <h4 className="font-semibold text-red-700">CAR OUT</h4>
          </div>
          
          {carOutEvent ? (
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                <CreditCard size={16} className="text-gray-500" />
                <span className="text-sm text-gray-600">RFID ID:</span>
                <span className="font-semibold">{carOutEvent.rfid_id}</span>
              </div>
              
              {carOutEvent.license_plate && carOutEvent.license_plate !== 'N/A' && (
                <div className="flex items-center space-x-2">
                  <Car size={16} className="text-gray-500" />
                  <span className="text-sm text-gray-600">License Plate:</span>
                  <span className="font-semibold">{carOutEvent.license_plate}</span>
                </div>
              )}
              
              <div className="flex items-center space-x-2">
                <Clock size={16} className="text-gray-500" />
                <span className="text-sm text-gray-600">
                  {new Date(carOutEvent.created_at).toLocaleString()}
                </span>
              </div>
            </div>
          ) : (
            <p className="text-gray-400 text-sm">No car exit detected</p>
          )}
        </div>
      </div>
    </div>
  )
}

